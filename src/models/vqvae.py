import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, Quantizer, QuantizerEMA

class vqvae(Record):
    def __init__(self, filter_nr, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size, EMA=True, latent_weight=0.25):

        self.emb_dim = emb_dim
        self.nr_emb = nr_emb
        self.latent_weight = latent_weight
        #encoder
        self.e_top = Encoder(n_resblock, filter_nr, kernel_size, downsample=2)
        self.e_bot = Encoder(n_resblock, filter_nr, kernel_size, downsample=4)

        # adapt filternr before quantizer
        self.xt_rs = tf.keras.layers.Conv2D(self.emb_dim, kernel_size=1, padding="same")
        self.xb_rs = tf.keras.layers.Conv2D(self.emb_dim, kernel_size=1, padding="same")

        # upsample quanitzed top to concatenate with quantized bottom
        self.qxt_up2 = tf.keras.layers.Conv2DTranspose(filter_nr, kernel_size=4, strides=2, padding="same")

        # quantizer
        if EMA:
            self.q_top = QuantizerEMA(nr_emb)
            self.q_bot = QuantizerEMA(nr_emb)
        else:
            self.q_top = Quantizer(nr_emb)
            self.q_bot = Quantizer(nr_emb)

        # decoder
        self.d_b = Decoder(inpt_dim, n_resblock, filter_nr, kernel_size, upsample=4)
        self.d_t = Decoder(filter_nr, n_resblock, filter_nr, kernel_size, upsample=2)

    def __call__(self, inpt, training=False):

        #TODO: check why others used permutations and concatinations for combining top/bottom
        # encode/downsample
        xb = self.e_bot(inpt) #[b, w/4, h/4, filters]
        xt= self.e_top(xb) #[b, w/8, h/8, filters]

        #### quantize
        ## top
        xt = self.xt_rs(xt) # kernel=1 conv to adjust filter number [b, w/8, h/8, emb_dim]
        qxt, l_t, perplexity_t = self.q_top(xt, training=training) # quantization: [b, w/8, h/8, emb_dim]
        dqxt = self.d_t(qxt) # [b, w/4, h/4, filters] # decode/upsample
        ## bot
        qxb, l_b, perplexity_b = self.q_bot(self.xb_rs(tf.concat([xb,dqxt],-1)), training=training) # [b, w/4, h/4, emb_dim]

        # decode/upsample
        qxt_up = self.qxt_up2(qxt) # bring to same size as qxb [b, w/4, h/4, emb_dim]
        img = self.d_b(tf.concat([qxb, qxt_up],-1))

        loss_latent = tf.reduce_mean(l_t) + tf.reduce_mean(l_b)
        loss_mse =  tf.reduce_mean(tf.square(inpt-img))#reconstruction loss
        loss = self.latent_weight*loss_latent + loss_mse

        return {"inpt":inpt,
                "img":tf.clip_by_value(img,0,1),
                "loss": loss,
                "loss_mse": loss_mse,
                "loss_latent": loss_latent,
                "perplexity_t": perplexity_t,
                "perplexity_b": perplexity_b,
                "qxt":qxt,
                "qxb":qxb,}



class Encoder(Record):
    def __init__(self, n_resblock=2, filters=256, kernel_size=4, downsample=2):

        if downsample == 2:
            self.layers = [tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                           tf.keras.layers.ReLU(),]
        elif downsample == 4:
            self.layers = [tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                            tf.keras.layers.ReLU(),
                            tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                            tf.keras.layers.ReLU(),]

        for _ in range(n_resblock):
            self.layers.append(ResBlock(filters))

    def __call__(self, x):
        with tf.name_scope("Encoder") as scope:
            for l in self.layers:
                x = l(x)
            return x


class Decoder(Record):
    def __init__(self, out_filter, n_resblock=2, nr_filters=256, kernel_size=4, upsample=2):
        #conv to adjust filter number: [b, w/8, h/8, filters] so residual blocks work
        self.layers = [tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")]
        self.layers.extend([ResBlock(nr_filters) for _ in range(n_resblock)])

        if upsample == 2:
            self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter,
                                                               kernel_size=4,
                                                               strides=2,
                                                               padding="same"))
        elif upsample == 4:
            self.layers.append(tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size=4, strides=2, padding="same"))
            self.layers.append(tf.keras.layers.ReLU())
            self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4, strides=2, padding="same"))

    def __call__(self, x):
        with tf.name_scope("Decoder") as scope:
            for l in self.layers:
                x = l(x)
            return x
