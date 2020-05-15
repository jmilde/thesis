import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block, sigmoid

class INTROVAE(tf.keras.Model):
    def __init__(self, inpt_dim, cond_dim, channels, btlnk, batch_size,
                 normalizer_enc=tf.keras.layers.BatchNormalization,
                 normalizer_dec=tf.keras.layers.BatchNormalization,
                 name="Introvae",
                 weight_rec=1,
                 weight_kl=1,
                 weight_neg = 1,
                 m_plus = 120,
                 lr_enc= 0.0002,
                 lr_dec= 0.0002,
                 beta1 = 0.5,
                 **kwargs):
        super(INTROVAE, self).__init__(name=name, **kwargs)

        self.inpt_dim   = inpt_dim
        self.btlnk      = btlnk
        self.batch_size = batch_size
        self.weight_rec = weight_rec
        self.weight_kl  = weight_kl
        self.weight_neg = weight_neg
        self.m_plus     = m_plus
        # encoding
        self.inpt_layer = tf.keras.layers.InputLayer(inpt_dim)
        self.inpt_layer_cond = tf.keras.layers.InputLayer(cond_dim)
        self.encoder    = Encoder(channels, btlnk, normalizer=normalizer_enc)

        # decoding
        self.decoder= Decoder(inpt_dim, channels[::-1], normalizer=normalizer_dec)

        # optimizers
        self.optimizer_enc= tf.keras.optimizers.Adam(lr_enc)#, beta_1=beta1) # todo: test beta1=0.5
        self.optimizer_dec= tf.keras.optimizers.Adam(lr_dec)#, beta_1=beta1)

    def encode(self, x, training=False):
        z, mu, lv = self.encoder(x)
        return z, mu, lv

    def decode(self, x, training=False):
        return self.decoder(x)

    def kl_loss(self, mu, lv):
        return tf.reduce_mean(0.5 * (tf.square(mu) + tf.exp(lv) - lv - 1.0))

    def mse_loss(self, x, x_rec):
        return  tf.reduce_mean(tf.reduce_sum(tf.math.square(x-x_rec), -1))


    def vae_step(self, x, training=True):
        x         = self.inpt_layer(x)
        z, mu, lv = self.encode(x, training=training)  # encode real image
        x_rec     = self.decode(z, z_cond, training=training)  # reconstruct real image

        # kl loss
        loss_kl = self.kl_loss(mu, lv)

        # reconstruction loss
        loss_rec =  self.mse_loss(x, x_rec)

        loss = loss_rec+loss_kl

        return {"x":x,
                "x_rec": x_rec,
                "loss": loss,
                "loss_kl": loss_kl,
                "loss_rec": loss_rec}


    @tf.function
    def train_vae(self, x):
        with tf.GradientTape() as tape:
            output = self.vae_step(x, training=True)
            loss = output["loss"]
        self.optimizer_enc.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return output

    def call(self, x, training=True):
        x     = self.inpt_layer(x[0])
        z_cond  = self.inpt_layer_cond(x[1])
        noise = tf.random.normal((self.batch_size, self.btlnk), 0, 1)#

        z, mu, lv = self.encode(x, training=training)  # encode real image
        x_rec     = self.decode(z, z_cond, training=training)  # reconstruct real image
        _, mu_rec, lv_rec = self.encode(tf.stop_gradient(x_rec)) # encode reconstruction

        x_fake = self.decode(noise, z_cond, training=training) # generate fake from noise
        _, mu_fake, lv_fake = self.encode(tf.stop_gradient(x_fake)) # encode fake


        ########
        # LOSS #
        ########
        # kl loss
        kl_real = self.kl_loss(mu, lv)
        kl_rec  = self.kl_loss(mu_rec, lv_rec)
        kl_fake = self.kl_loss(mu_fake, lv_fake)

        # reconstruction loss
        loss_rec =  self.mse_loss(x, x_rec)

        # margin loss
        loss_margin = kl_real + 0.5*(tf.keras.layers.ReLU(self.m_plus - kl_rec) + tf.keras.layers.ReLU(self.m_plus-kl_fake) ) *self.weight_neg

        loss_enc = loss_rec * self.weight_rec + loss_margin * self.weight_kl
        loss_dec = 0.5*(kl_rec + kl_fake) * self.weight_kl
        return {"x"        : x,
                "x_rec"    : x_rec,
                "x_fake"   : x_fake,
                "loss_enc" : loss_enc,
                "loss_dec" : loss_dec,
                "loss_rec" : loss_rec,
                "kl_real"  : kl_real,
                "kl_fake"  : kl_fake,
                "kl_rec"   : kl_rec,
                "mu"       : mu,
                "lv"       : lv,
                "mu_rec"   : mu_rec,
                "lv_rec"   : lv_rec,
                "mu_fake"  : mu_fake,
                "lv_fake"  : lv_fake}


    @tf.function
    def train(self, x):
        with tf.GradientTape() as e_tape, tf.GradientTape() as d_tape:
            output = self.call(x, training = True)
            e_loss = output["loss_enc"] # encoder
            d_loss = output["loss_dec"] # decoder
        self.optimizer_enc_gen.apply_gradients(zip(e_tape.gradient(e_loss, self.encoder.trainable_variables      ), self.encoder.trainable_variables))
        self.optimizer_dec.apply_gradients(    zip(d_tape.gradient(d_loss, self.decoder.trainable_variables), self.decoder.trainable_variables))
        return output




class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 latent_dim,
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
            self.layers.append(ResBlock(channel,
                                        adjust_channels=channel!=channels[i-1]))
            self.layers.append(tf.keras.layers.AveragePooling2D())

        # additional res layer
        self.layers.append(ResBlock(channels[-1]))

        self.variational = VariationalEncoding(latent_dim)

    def call(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training=training)
        z, mu, lv = self.variational(x, training=training)
        return z, mu, lv



class Decoder(tf.keras.layers.Layer):
    def __init__(self, inpt_dim,
                 channels=[512, 512, 512, 256, 128, 64],
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        xy_dim = inpt_dim[1]/(2**len(channels)) #wh
        self.dec_reshape_dim = (-1, int(xy_dim), int(xy_dim), channels[0])

        # dense cond
        self.dense_cond = tf.keras.layers.Dense(channels[0],name="dense_resize")

        # dense layer to resize and reshape input
        self.dense_resize = tf.keras.layers.Dense(xy_dim*xy_dim*channels[0],name="dense_resize")
        self.relu = tf.keras.layers.ReLU()

        self.layers = []
        self.layers.append(ResBlock(channels[0]))
        self.layers.append(tf.keras.layers.UpSampling2D(size=(2,2)))

        for i,channel in enumerate(channels[1:], 1):
            self.layers.append(ResBlock(channel,
                                        adjust_channels= channel!=channels[i-1]))
            self.layers.append(tf.keras.layers.UpSampling2D(size=(2,2)))

        # additional res layer
        self.layers.append(ResBlock(channels[-1]))

        self.layers.append(tf.keras.layers.Conv2D(inpt_dim[-1], kernel_size=5, strides=1,
                                                      use_bias = False,
                                                      padding="same"))

    def call(self, x, cond, training=False):
        cond = self.relu(self.dense_cond(cond))
        x = tf.reshape(self.relu(self.dense_resize(tf.concat(x, cond))), self.dec_reshape_dim)
        for layer in self.layers:
            x = layer(x, training=training)
        return x
