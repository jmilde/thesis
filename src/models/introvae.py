import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block, sigmoid

class INTROVAE(tf.keras.Model):
    def __init__(self, inpt_dim, channels, btlnk, batch_size, cond_dim_colors,
                 rnn_dim, cond_dim_txts, vocab_dim, emb_dim,
                 normalizer_enc           = tf.keras.layers.BatchNormalization,
                 normalizer_dec           = tf.keras.layers.BatchNormalization,
                 name                     = "Introvae",
                 weight_rec               = 1,
                 weight_kl                = 1,
                 weight_neg               = 1,
                 m_plus                   = 100,
                 lr_enc                   = 0.0002,
                 lr_dec                   = 0.0002,
                 beta1                    = 0.5,
                 noise_color              = 0.1,
                 noise_txt                = 0.1,
                 noise_img                = 0.1,
                 dropout_conditionals     = 0.3,
                 dropout_encoder_resblock = 0.3,
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
        self.inpt_layer       = tf.keras.layers.InputLayer(inpt_dim)
        self.inpt_layer_cond  = tf.keras.layers.InputLayer(input_shape=(None,None))
        self.inpt_layer_txt   = tf.keras.layers.InputLayer(input_shape=(None,None))
        self.encoder          = Encoder(channels, btlnk, normalizer=normalizer_enc, dropout_rate=dropout_encoder_resblock)
        self.noise_img        = tf.keras.layers.GaussianNoise(noise_img)

        # decoding
        self.decoder= Decoder(inpt_dim, channels[::-1], rnn_dim,
                              vocab_dim, emb_dim, cond_dim_colors,
                              cond_dim_txts, dropout_conditionals,
                              noise_color, noise_txt, normalizer=normalizer_dec)

        self.relu = tf.keras.layers.ReLU()
        # optimizers
        self.optimizer_enc= tf.keras.optimizers.Adam(lr_enc, beta_1=beta1)
        self.optimizer_dec= tf.keras.optimizers.Adam(lr_dec, beta_1=beta1)

    def encode(self, x, training=False):
        z, mu, lv = self.encoder(x, training=training)
        return z, mu, lv

    def decode(self, x, colors, txts, training=False):
        return self.decoder(x, colors, txts, training=training)

    def kl_loss(self, mu, lv):
        return tf.reduce_mean(-0.5 * tf.reduce_sum((-1*(tf.square(-mu)+tf.exp(lv))+1+lv),-1))+0.0

    def mse_loss(self, x, x_rec, size_average=True):
        x = tf.reduce_sum(tf.math.square(tf.reshape(x-x_rec,(x.shape[0],-1))), -1)
        if size_average:
            return tf.reduce_mean(x)+0.0
        else:
            return tf.reduce_sum(x)+0.0

    def generate(self, x, color, spm_txt, training=False):
        x = self.noise_img(x, training=training)
        return self.decode(x, color, spm_txt, training=training)

    def vae_step(self, x, colors, txts, training=True):
        x         = self.noise_img(self.inpt_layer(x), training=training)
        x_color   = self.inpt_layer_cond(colors)
        x_txt     = self.inpt_layer_txt(txts)

        # encode
        z, mu, lv = self.encode(x, training=training)  # encode real image

        # decode
        #x_rec = self.decode(tf.concat([z, cond_color, cond_txts], 1), training=training)  # reconstruct real image
        x_rec = self.decode(z, x_color, x_txt, training=training)
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
    # maybe with tf 2.2 this works
    #input_signature=[tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
                                  #tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                                  #tf.TensorSpec(shape=(None,None), dtype=tf.int64)])
    @tf.function(experimental_relax_shapes=True)
    def train_vae(self, x, colors, txts):
        with tf.GradientTape() as tape:
            output = self.vae_step(x, colors, txts, training=True)
            loss = output["loss"]
        self.optimizer_enc.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return output


    def call(self, x, colors, txts, training=True):
        # inpts
        x         = self.inpt_layer(x)
        x_color   = self.inpt_layer_cond(colors)
        x_txt     = self.inpt_layer_txt(txts)
        z_p       = tf.random.normal((self.batch_size, self.btlnk), 0, 1)

        #########


        z, mu, lv = self.encode(x, training=training)  # encode real image
        x_r       = self.decode(z, x_color, x_txt, training=training)  # reconstruct real image
        x_p       = self.decode(z_p, x_color, x_txt, training=training) # generate fake from z_p

        # reconstruction loss
        loss_rec =  self.mse_loss(x, x_r)


        # no gradient flow for encoder
        _, mu_r_, lv_r_ = self.encode(tf.stop_gradient(x_r), training=training) # encode reconstruction
        _, mu_p_, lv_p_ = self.encode(tf.stop_gradient(x_p), training=training) # encode fake

        # Encoder Adversarial Loss
        kl_real  = self.kl_loss(mu, lv)
        kl_rec_  = self.kl_loss(mu_r_, lv_r_)
        kl_fake_ = self.kl_loss(mu_p_, lv_p_)
        loss_enc_adv = kl_real + 0.5*(self.relu(self.m_plus - kl_rec_) + self.relu(self.m_plus-kl_fake_) ) *self.weight_neg


        ### ENCODER LOSS
        loss_enc = loss_rec * self.weight_rec + loss_enc_adv * self.weight_kl


        # gradient flow for decoder
        _, mu_r, lv_r = self.encode(x_r, training=training) # encode reconstruction
        _, mu_p, lv_p = self.encode(x_p, training=training) # encode fake

        ### DECODER
        kl_rec  = self.kl_loss(mu_r, lv_r)
        kl_fake = self.kl_loss(mu_p, lv_p)
        loss_dec_adv = 0.5*(kl_rec + kl_fake)
        loss_dec = loss_dec_adv * self.weight_kl + loss_rec * self.weight_rec


        return {"x"        : x,
                "x_r"      : x_r,
                "x_p"      : x_p,
                "loss_enc" : loss_enc,
                "loss_dec" : loss_dec,
                "loss_rec" : loss_rec,
                "kl_real"  : kl_real,
                "kl_rec"   : kl_rec,
                "kl_fake"  : kl_fake,
                "loss_enc_adv": loss_enc_adv,
                "loss_dec_adv": loss_dec_adv,
                "mu"       : tf.reduce_mean(mu),
                "lv"       : tf.reduce_mean(lv)}


    @tf.function(experimental_relax_shapes=True)
    def train(self, x, colors, txts,):
        with tf.GradientTape() as e_tape, tf.GradientTape() as d_tape:
            output = self.call(x, colors, txts, training = True)
            e_loss = output["loss_enc"] # encoder
            d_loss = output["loss_dec"] # decoder
        self.optimizer_enc.apply_gradients(zip(e_tape.gradient(e_loss, self.encoder.trainable_variables), self.encoder.trainable_variables))
        self.optimizer_dec.apply_gradients(zip(d_tape.gradient(d_loss, self.decoder.trainable_variables), self.decoder.trainable_variables))
        return output




class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 channels,
                 latent_dim,
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Encoder",
                 dropout_rate=0.2,
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
                                        adjust_channels=channel!=channels[i-1],
                                        dropout_rate=dropout_rate))
            self.layers.append(tf.keras.layers.AveragePooling2D())

        # additional res layer
        self.layers.append(ResBlock(channels[-1], dropout_rate=dropout_rate))

        self.variational = VariationalEncoding(latent_dim)

    def call(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training=training)
        z, mu, lv = self.variational(x, training=training)
        return z, mu, lv



class Decoder(tf.keras.layers.Layer):
    def __init__(self, inpt_dim,
                 channels,
                 rnn_dim,
                 vocab_dim,
                 emb_dim,
                 cond_dim_colors,
                 cond_dim_txts,
                 dropout_conditionals,
                 noise_color,
                 noise_txt,
                 normalizer=tf.keras.layers.BatchNormalization,
                 name="Decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.relu = tf.keras.layers.ReLU()

        xy_dim = inpt_dim[1]/(2**len(channels)) #wh
        self.dec_reshape_dim = (-1, int(xy_dim), int(xy_dim), channels[0])


        # dense layer to resize and reshape input
        self.RNN              = GRU_bidirectional(rnn_dim, vocab_dim, emb_dim)
        self.dense_cond_color = tf.keras.layers.Dense(cond_dim_colors, name="dense_cond_color")
        self.dense_cond_txt   = tf.keras.layers.Dense(cond_dim_txts, name="dense_cond_txt")
        self.dense_resize     = tf.keras.layers.Dense(xy_dim*xy_dim*channels[0],name="dense_resize")
        self.dropout_color    = tf.keras.layers.Dropout(dropout_conditionals)
        self.dropout_txt      = tf.keras.layers.Dropout(dropout_conditionals)
        self.noise_color      = tf.keras.layers.GaussianNoise(noise_color)
        self.noise_txt        = tf.keras.layers.GaussianNoise(noise_txt)
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

    def call(self, z, x_color, x_txt, training=False):
        cond_color = self.dropout_color(self.noise_color(self.relu(self.dense_cond_color(x_color)), training), training)
        cond_txts  = self.dropout_txt(self.noise_txt(self.relu(self.dense_cond_txt(self.RNN(x_txt))), training), training)
        x = tf.concat([z, cond_color, cond_txts], 1)

        x = self.relu(self.dense_resize(x))
        x = tf.reshape(x, self.dec_reshape_dim)

        for layer in self.layers:
            x = layer(x, training=training)
        return x #tf.keras.activations.sigmoid(x) # in orignial paper no sigmoid somehow


class GRU_bidirectional(tf.keras.layers.Layer):
    def __init__(self, rnn_dim,
                 vocab_dim,
                 emb_dim,
                 name="GRU_bidirectional",
                 **kwargs):
        super(GRU_bidirectional, self).__init__(name=name, **kwargs)

        self.emb_layer = tf.keras.layers.Embedding(input_dim=vocab_dim, output_dim=emb_dim)
        forward_layer = tf.keras.layers.GRU(rnn_dim)
        backward_layer = tf.keras.layers.GRU(rnn_dim, go_backwards=True)
        self.bidirectional = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(None, None, emb_dim))

    def call(self, x, training=False):
        x = self.emb_layer(x)
        x = self.bidirectional(x)
        return x
