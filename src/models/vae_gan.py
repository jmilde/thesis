import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block, sigmoid

class VAEGAN(tf.keras.Model):
    def __init__(self, inpt_dim, channels, btlnk, batch_size,
                 accelerate=1,
                 normalizer_enc=tf.keras.layers.BatchNormalization,
                 normalizer_dec=tf.keras.layers.BatchNormalization,
                 name="VAE",
                 **kwargs):
        super(VAEGAN, self).__init__(name=name, **kwargs)

        self.accelerate = accelerate
        self.inpt_dim   = inpt_dim
        self.btlnk      = btlnk
        self.batch_size = batch_size
        # encoding
        self.inpt_layer = tf.keras.layers.InputLayer(inpt_dim)
        self.encoder = Encoder(channels, btlnk, normalizer=normalizer_enc)

        # generating
        self.generator = Generator(inpt_dim, channels[::-1], normalizer=normalizer_dec)

        # descriminating
        self.discriminator = Discriminator(channels, normalizer=normalizer_enc)
        self.d_scale_factor = tf.constant(0.1)

        # optimizers
        self.optimizer_enc_gen= tf.keras.optimizers.Adam()#1e-3, beta_1=0.5)
        self.optimizer_dec= tf.keras.optimizers.Adam(self.get_lr_dec)#, beta_1=0.5)

    def encode(self, x, training=False):
        z, mu, lv = self.encoder(x)
        return z, mu, lv

    def generate(self, x, training=False):
        return self.generator(x)

    def discriminate(self, x, training=False):
        return self.discriminator(x)

    def get_lr_dec(self, lr_start=1e-3):
        return tf.constant(lr_start) * self.lr_balancer

    def call(self, x, step=0, training=True):
        x     = self.inpt_layer(x)
        noise = tf.random.normal((self.batch_size, self.btlnk), 0, 1)#

        z, mu, lv = self.encode(x, training=training)

        x_rec   = self.generate(z,     training=training)
        x_noise = self.generate(noise, training=training)

        # discriminator
        dx       = self.discriminate(x,       training=training)
        dx_rec   = self.discriminate(x_rec,   training=training)
        dx_noise = self.discriminate(x_noise, training=training)

        ########
        # LOSS #
        ########
        # kl loss
        rate = tf.cast(step, tf.float32)/self.accelerate
        rate_anneal = tf.tanh(rate)
        loss_latent = tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

        # reconstruction loss
        loss_rec =  tf.reduce_mean(tf.math.square(x-x_rec))

        # GAN loss
        gx_rec_loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx_rec), logits=dx_rec))
        gx_noise_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx_noise), logits=dx_noise))
        dx_loss       = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dx) - self.d_scale_factor, logits=dx))
        dx_rec_loss   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dx_rec), logits=dx_rec))
        dx_noise_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dx_noise), logits=dx_noise))


        d_loss = dx_loss + (dx_rec_loss + dx_noise_loss)/2
        g_loss = (gx_rec_loss + gx_noise_loss)/2 + loss_rec*10
        e_loss = loss_rec*10 + loss_latent

        self.lr_balancer = sigmoid(dx_rec_loss - gx_rec_loss, mult=10)

        return {"x":x,
                "x_rec": x_rec,
                "x_noise": x_noise,
                "d_loss": d_loss,
                "g_loss": g_loss,
                "e_loss": e_loss,
                "lr_balancer": self.lr_balancer,
                "gx_rec_loss": gx_rec_loss,
                "dx_loss": dx_loss,
                "dx_rec_loss": dx_rec_loss,
                "dx_noise_loss": dx_noise_loss,
                "loss_rec": loss_rec,
                "loss_latent": loss_latent,
                "mu":tf.reduce_mean(mu),
                "lv": tf.reduce_mean(lv),
                "rate_anneal": rate_anneal}


    @tf.function
    def train(self, x, step):
        with tf.GradientTape() as e_tape, tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            output = self.call(x, step, training = True)
            e_loss = output["e_loss"] # encoder
            g_loss = output["g_loss"] # generator
            d_loss = output["d_loss"] # discriminator
        self.optimizer_enc_gen.apply_gradients(zip(e_tape.gradient(e_loss, self.encoder.trainable_variables      ), self.encoder.trainable_variables))
        self.optimizer_enc_gen.apply_gradients(zip(g_tape.gradient(g_loss, self.generator.trainable_variables    ), self.generator.trainable_variables))
        self.optimizer_dec.apply_gradients(    zip(d_tape.gradient(d_loss, self.discriminator.trainable_variables), self.discriminator.trainable_variables))
        return output


class Discriminator(tf.keras.layers.Layer):
    def __init__(self, channels=[64, 128, 256, 512, 512, 512],
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Discriminator",
                 **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)

        self.layers = [tf.keras.layers.Conv2D(channels[0],
                                         kernel_size=5,
                                         strides=1,
                                         padding="same",
                                         use_bias=False,
                                         activation="relu"),
                  normalizer(),
                  tf.keras.layers.ReLU(),
                  tf.keras.layers.AveragePooling2D()]

        for i, channel in enumerate(channels[1:], 1):
            adjust = channel!=channels[i-1]
            self.layers.append(ResBlock(channel, adjust_channels=adjust))
            self.layers.append(tf.keras.layers.AveragePooling2D())


        # additional residual layer
        self.layers.append(ResBlock(channels[-1]))

        self.layers.append(tf.keras.layers.Flatten(data_format="channels_last"))
        self.layers.append(tf.keras.layers.Dense(512, activation="relu", use_bias=False))
        self.layers.append(tf.keras.layers.Dense(1, use_bias=False))

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, channels,
                 btlnk,
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Encoder",
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.layers = [tf.keras.layers.Conv2D(channels[0],
                                         kernel_size=5,
                                         strides=1,
                                         padding="same",
                                         use_bias=False,
                                         activation="relu"),
                  normalizer(),
                  tf.keras.layers.ReLU(),
                  tf.keras.layers.AveragePooling2D()]

        for i, channel in enumerate(channels[1:], 1):
            adjust = channel!=channels[i-1]
            self.layers.append(ResBlock(channel, adjust_channels=adjust))
            self.layers.append(tf.keras.layers.AveragePooling2D())

        # additional res layer
        self.layers.append(ResBlock(channels[-1]))

        self.variational = VariationalEncoding(btlnk)

    def call(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training=training)
        z, mu, lv = self.variational(x, training=training)
        return z, mu, lv



class Generator(tf.keras.layers.Layer):
    def __init__(self, inpt_dim,
                 channels=[512, 512, 512, 256, 128, 64],
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Generator",
                 **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)

        # dense layer to resize and reshape input
        xy_dim = inpt_dim[1]/(2**len(channels)) #wh
        self.dec_reshape_dim = (-1, int(xy_dim), int(xy_dim), channels[0])
        self.dense_resize = tf.keras.layers.Dense(xy_dim*xy_dim*channels[0],name="dense_resize")

        self.layers = []
        for channel in channels:
            self.layers.append(tf.keras.layers.Conv2DTranspose(channel,
                                                          kernel_size=4,
                                                          strides=2,
                                                          padding="same",
                                                          activation="relu"))

        # additional res layer
        self.layers.append(ResBlock(channels[-1]))

        self.layers.append(tf.keras.layers.Conv2DTranspose(inpt_dim[-1], kernel_size=4, strides=1,
                                                      use_bias = False,
                                                      padding="same", activation="sigmoid"))

    def call(self, x, training=False):
        x = tf.reshape(self.dense_resize(x), self.dec_reshape_dim)
        for layer in self.layers:
            x = layer(x, training=training)
        return x
