import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_conv_block, upsampling_conv_block

class vae(Record):
    def __init__(self, n_downsampling, btlnk, filter_nr, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size,
                 accelerate=1e-4,
                 normalizer_enc=tf.keras.layers.BatchNormalization,
                 normalizer_dec=tf.keras.layers.BatchNormalization,):
        self.accelerate = accelerate

        self.encoder = Encoder(n_downsampling, n_resblock, filter_nr, kernel_size, normalizer=normalizer_enc)
        self.variational = VariationalEncoding(btlnk)
        self.resize  = tf.keras.layers.Dense(6*6*256) #TODO: flexible: size = first deconvolution(b,w,c)
        self.decoder = Decoder(n_downsampling, inpt_dim, n_resblock, filter_nr, kernel_size, normalizer=normalizer_dec)

    def __call__(self, inpt, step=0):
        x = self.encoder(inpt)
        z, mu, lv = self.variational(x)
        x = tf.reshape(self.resize(z), (-1, 6, 6, 256)) #TODO: flexible
        img = self.decoder(x)

        # kl loss
        rate = self.accelerate * tf.cast(step, tf.float32)
        rate_anneal = tf.tanh(rate)
        loss_kl = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

        # TODO: try different loss functions
        #epsilon = 1e-10
        #loss_rec = tf.reduce_mean(-tf.reduce_sum(inpt * tf.math.log(epsilon+img) + (1-inpt) * tf.math.log(epsilon+1-img),  axis=1))
        loss_rec =  tf.reduce_mean(tf.square(inpt-img)) #mean squared error
        #loss_rec = tf.reduce_mean(tf.abs(inpt - img))  # absolute loss

        loss = loss_rec + loss_kl*rate_anneal

        return {"inpt":inpt,
                "img":tf.clip_by_value(img,0,1),
                "loss": loss,
                "loss_rec": loss_rec,
                "loss_kl": loss_kl,
                "mu":tf.reduce_mean(mu),
                "lv": tf.reduce_mean(lv),}


class Encoder(Record):
    def __init__(self,n_downsampling, n_resblock=2, filters=256, kernel_size=4, normalizer=tf.keras.layers.BatchNormalization):
    # downsample by 16 -> [b,24,24,3]
        self.layers = [downsampling_conv_block(filters, normalizer, kernel_size=kernel_size) for _ in range(n_downsampling)]

        for _ in range(n_resblock):
            self.layers.append(ResBlock(filters, normalizer))

    def __call__(self, x):
        with tf.name_scope("Encoder") as scope:
            for l in self.layers:
                x = l(x)
        return x


class Decoder(Record):
    def __init__(self,n_upsampling, out_filter, n_resblock=2, nr_filters=256, kernel_size=4, upsample=2, normalizer=tf.keras.layers.BatchNormalization):
        #conv to adjust filter number: [b, w/8, h/8, filters] so residual blocks work
        self.layers = [tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")]
        self.layers.extend([ResBlock(nr_filters, normalizer) for _ in range(n_resblock)])
        self.layers.extend([upsampling_conv_block(nr_filters, normalizer, kernel_size=kernel_size) for _ in range(n_upsampling-1)])
        self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4,strides=2, padding="same"))
        self.layers.append(normalizer())
    def __call__(self, x):
        with tf.name_scope("Decoder") as scope:
            for l in self.layers:
                x = l(x)
            return x
