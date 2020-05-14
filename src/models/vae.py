import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block

class VAE(tf.keras.Model):
    def __init__(self, inpt_dim, channels, btlnk,
                 accelerate=1,
                 normalizer_enc=tf.keras.layers.BatchNormalization,
                 normalizer_dec=tf.keras.layers.BatchNormalization,
                 optimizer= tf.keras.optimizers.Adam(),
                 name="VAE",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)

        self.accelerate = accelerate
        self.inpt_dim = inpt_dim
        self.optimizer = optimizer

        self.inpt_layer = tf.keras.layers.InputLayer(inpt_dim)
        self.encoder = Encoder(channels, normalizer=normalizer_enc)
        self.variational = VariationalEncoding(btlnk)

        xy_dim_after_downsample = inpt_dim[1]/(2**len(channels))
        self.dec_reshape_dim = (-1, int(xy_dim_after_downsample), int(xy_dim_after_downsample), channels[-1])
        self.dense_resize  = tf.keras.layers.Dense(xy_dim_after_downsample*xy_dim_after_downsample*channels[-1],name="dense_resize")
        self.decoder = Decoder(inpt_dim[-1], channels[::-1], normalizer=normalizer_dec)

    def encode(self, x, training=True):
        x = self.encoder(x, training=training)
        z, mu, lv = self.variational(x)
        return z, mu, lv

    def decode(self, x, training=True):
        return self.decoder(x)

    def call(self, x, step=0, training=True):
        x = self.inpt_layer(x)

        z, mu, lv = self.encode(x, training=training)
        z_reshape = tf.reshape(self.dense_resize(z), self.dec_reshape_dim)
        x_rec = self.decode(z_reshape, training=training)

        # kl loss
        rate = tf.cast(step, tf.float32)/self.accelerate
        rate_anneal = tf.tanh(rate)
        loss_latent = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

        # reconstruction loss
        loss_rec =  tf.reduce_mean(tf.math.square(x-x_rec))


        loss = loss_rec + loss_latent*rate_anneal

        return {"x":x,
                "x_rec": x_rec,
                "loss": loss,
                "loss_rec": loss_rec,
                "loss_latent": loss_latent,
                "mu":tf.reduce_mean(mu),
                "lv": tf.reduce_mean(lv),
                "rate_anneal": rate_anneal}


    @tf.function
    def train(self, x, step):
        with  tf.GradientTape() as tape:
            output = self.call(x, step, training = True)
            loss = output["loss"]
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 channels=[64, 128, 256, 512, 512, 512],
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Encoder",
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.layers = [tf.keras.layers.Conv2D(channels[0],
                                         kernel_size=5,
                                         strides=1,
                                         padding="same",
                                         use_bias=False),
                  normalizer(),
                  tf.keras.layers.ReLU(),
                  tf.keras.layers.AveragePooling2D()]

        for i, channel in enumerate(channels[1:], 1):
            adjust = channel!=channels[i-1]
            self.layers.append(ResBlock(channel, adjust_channels=adjust))
            self.layers.append(tf.keras.layers.AveragePooling2D())


        self.layers.append(ResBlock(channels[-1]))


    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x



class Decoder(tf.keras.layers.Layer):
    def __init__(self, out_channels,
                 channels=[512, 512, 512, 256, 128, 64],
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.layers = []

        #for i, channel in enumerate(channels):
         #   adjust = channel!=channels[max(0,i-1)]
          #  self.layers.append(ResBlock(channel, adjust_channels=adjust))
           # self.layers.append(tf.keras.layers.UpSampling2D())

        #self.layers.append(ResBlock(channels[-1]))

        for channel in channels:
            self.layers.append(tf.keras.layers.Conv2DTranspose(channel,
                                                          kernel_size=4,
                                                          strides=2,
                                                          padding="same",
                                                          activation="relu"))

        self.layers.append(tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=4, strides=1,
                                                      use_bias = False,
                                                      padding="same", activation="sigmoid"))

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x
