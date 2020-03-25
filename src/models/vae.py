import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block

class vae(Record):
    def __init__(self, n_downsampling, filter_startsize, btlnk, res_filters, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size,
                 accelerate=1e-4,
                 img_dim = (384,384,3),
                 normalizer_enc=tf.keras.layers.BatchNormalization,
                 normalizer_dec=tf.keras.layers.BatchNormalization,):
        self.accelerate = accelerate

        self.encoder = Encoder(n_downsampling, filter_startsize, n_resblock, res_filters, kernel_size, normalizer=normalizer_enc)
        self.variational = VariationalEncoding(btlnk)

        dec_first_conv_filters = filter_startsize*(2**n_downsampling)
        xy_dim_after_downsample = img_dim[1]/(2**n_downsampling)
        self.dec_reshape_dim = (-1, int(xy_dim_after_downsample), int(xy_dim_after_downsample), int(dec_first_conv_filters))
        self.resize  = tf.keras.layers.Dense(xy_dim_after_downsample*xy_dim_after_downsample*dec_first_conv_filters)
        self.decoder = Decoder(n_downsampling, inpt_dim, dec_first_conv_filters, n_resblock, res_filters, kernel_size, normalizer=normalizer_dec)

    def __call__(self, inpt, step=0):
        x = self.encoder(inpt)
        z, mu, lv = self.variational(x)
        x = tf.reshape(self.resize(z), self.dec_reshape_dim)
        img = self.decoder(x)
        img = tf.math.sigmoid(img)

        # kl loss
        rate = self.accelerate * tf.cast(step, tf.float32)
        rate_anneal = tf.tanh(rate)
        loss_kl = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

        # TODO: try different loss functions
        #epsilon = 1e-10
        #loss_rec = tf.reduce_mean(-tf.reduce_sum(inpt * tf.math.log(epsilon+img) + (1-inpt) * tf.math.log(epsilon+1-img),  axis=1))
        loss_rec =  tf.reduce_mean(tf.reduce_sum(tf.math.square(inpt-img), axis=0))
        #mean squared error
        #loss_rec = tf.reduce_mean(tf.abs(inpt - img))  # absolute loss

        loss = loss_rec + loss_kl*rate_anneal

        return {"inpt":inpt,
                "img": img,
                #"img":tf.clip_by_value(img,0,1),
                "loss": loss,
                "loss_rec": loss_rec,
                "loss_kl": loss_kl,
                "mu":tf.reduce_mean(mu),
                "lv": tf.reduce_mean(lv),}


class Encoder(Record):
    def __init__(self,n_downsampling, filter_startsize=32, n_resblock=2, res_filters=256,  kernel_size=4, normalizer=tf.keras.layers.BatchNormalization):
    # downsample by 16 -> [b,24,24,3]
        #self.layers = [downsampling_res_block(filter_startsize*(2**i), normalizer, kernel_size=kernel_size) for i, _ in enumerate(range(n_downsampling))]
        self.layers = [downsampling_res_block(filter_startsize*(2**i), normalizer, kernel_size=kernel_size) for i, _ in enumerate(range(n_downsampling))]

        #for _ in range(n_resblock):
        #    self.layers.append(ResBlock(res_filters, normalizer))

    def __call__(self, x):
        with tf.name_scope("Encoder") as scope:
            for l in self.layers:
                x = l(x)
        return x


class Decoder(Record):
    def __init__(self,n_upsampling, out_filter, filter_startsize=2048, n_resblock=2, res_filters=256, nr_filters=256, kernel_size=4, upsample=2, normalizer=tf.keras.layers.BatchNormalization):
        #conv to adjust filter number: [b, w/8, h/8, filters] so residual blocks work

        self.layers= [tf.keras.layers.Conv2DTranspose(int(filter_startsize/(2**i)), kernel_size=4, strides=2, padding="same")
                           for i, _ in enumerate(range(n_upsampling))]
        #self.layers = [tf.keras.layers.Conv2D(res_filters, kernel_size=3, padding="same")]
        #self.layers.extend([ResBlock(res_filters, normalizer) for _ in range(n_resblock)])
        #self.layers.extend([upsampling_res_block(filter_startsize/(2**i), normalizer, kernel_size=kernel_size) for i, _ in enumerate(range(n_upsampling)]))
        self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4, strides=1, padding="same"))
        #self.layers.append(normalizer())
    def __call__(self, x):
        with tf.name_scope("Decoder") as scope:
            for l in self.layers:
                x = l(x)
            return x
