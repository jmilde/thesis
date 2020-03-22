import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding

class vae(Record):
    def __init__(self, btlnk, filter_nr, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size, accelerate=1e-4):
        self.accelerate = accelerate

        self.encoder = Encoder(n_resblock, filter_nr, kernel_size)
        self.variational = VariationalEncoding(btlnk)
        self.resize  = tf.keras.layers.Dense(6*6*3) #TODO: flexible #img_x/8, img_y/8, channels
        self.decoder = Decoder(inpt_dim, n_resblock, filter_nr, kernel_size)

    def __call__(self, inpt, step=0):
        x = self.encoder(inpt)
        z, mu, lv = self.variational(x)
        x = tf.reshape(self.resize(z), (-1, 6, 6, 3)) #TODO: flexible
        img = self.decoder(x)

        # kl loss
        rate = self.accelerate * tf.cast(step, tf.float32)
        rate_anneal = tf.tanh(rate)
        loss_kl = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

        # TODO: try different loss functions
        epsilon = 1e-10o
        #loss_rec = tf.reduce_mean(-tf.reduce_sum(inpt * tf.math.log(epsilon+img) + (1-inpt) * tf.math.log(epsilon+1-img),  axis=1))
        #loss_rec =  tf.reduce_mean(tf.square(inpt-img)) #mean squared error
        loss_rec = tf.reduce_mean(tf.abs(inpt - img))  # absolute loss

        loss = loss_rec + loss_kl*rate_anneal

        return {"inpt":inpt,
                "img":tf.clip_by_value(img,0,1),
                "loss": loss,
                "loss_rec": loss_rec,
                "loss_kl": loss_kl,
                "mu":tf.reduce_mean(mu),
                "lv": tf.reduce_mean(lv),}



class Encoder(Record):
    def __init__(self, n_resblock=2, filters=256, kernel_size=4):
    # downsample by 16 -> [b,24,24,3]
        self.relu = tf.keras.layers.ReLU()
        self.layers = [tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,
                       tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,
                       tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,
                       tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,
                       tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,
                       tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, padding="same"),
                       self.relu,]

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
        self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4,strides=2, padding="same"))
        self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size=4, strides=2, padding="same"))
        self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size=4, strides=2, padding="same"))
        self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size=4, strides=2, padding="same"))
        self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(tf.keras.layers.Conv2DTranspose(nr_filters, kernel_size=4, strides=2, padding="same"))
        self.layers.append(tf.keras.layers.ReLU())
        self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4, strides=2, padding="same"))

    def __call__(self, x):
        with tf.name_scope("Decoder") as scope:
            for l in self.layers:
                x = l(x)
            return x
