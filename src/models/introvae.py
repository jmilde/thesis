import tensorflow as tf
from src.util import Record
from src.util_tf import ResBlock, VariationalEncoding, downsampling_res_block, upsampling_res_block, sigmoid

class INTROVAE(tf.keras.Model):
    def __init__(self, inpt_dim, channels, btlnk, batch_size, cond_dim_colors,
                 rnn_dim, cond_dim_txts, cond_dim_clusters, vocab_dim, emb_dim,
                 color_cond_dim, txt_cond_dim, cluster_cond_dim,
                 color_cond_type, txt_cond_type, cluster_cond_type,
                 cond_model               = None,
                 normalizer_enc           = tf.keras.layers.BatchNormalization,
                 normalizer_dec           = tf.keras.layers.BatchNormalization,
                 weight_rec               = 1,
                 weight_kl                = 1,
                 weight_neg               = 1,
                 weight_aux               = 1,
                 m_plus                   = 100,
                 lr_enc                   = 0.0002,
                 lr_dec                   = 0.0002,
                 beta1                    = 0.5,
                 beta2                    = 0.999,
                 noise_color              = 0,
                 noise_txt                = 0,
                 noise_img                = 0,
                 dropout_conditionals     = 0,
                 dropout_encoder_resblock = 0,
                 auxilary                 = False,
                 name                     = "Introvae",
                 **kwargs):
        super(INTROVAE, self).__init__(name=name, **kwargs)

        self.auxilary            = auxilary
        self.cond_model          = cond_model
        self.color_cond_type     = color_cond_type
        self.txt_cond_type       = txt_cond_type
        self.cluster_cond_type   = cluster_cond_type
        self.inpt_dim            = inpt_dim
        self.btlnk               = btlnk
        self.batch_size          = batch_size
        self.weight_rec          = weight_rec
        self.weight_kl           = weight_kl
        self.weight_neg          = weight_neg
        self.weight_aux          = weight_aux
        self.m_plus              = m_plus

        # encoding
        self.inpt_layer          = tf.keras.layers.InputLayer(inpt_dim)
        self.inpt_layer_color    = tf.keras.layers.InputLayer(input_shape=(None,None))
        self.inpt_layer_cluster  = tf.keras.layers.InputLayer(input_shape=(None,None))
        self.inpt_layer_txt      = tf.keras.layers.InputLayer(input_shape=(None,None))
        self.conditional_embedder = Conditional_Embedder(color_cond_type,
                                                         txt_cond_type,
                                                         cluster_cond_type,
                                                         cond_dim_colors,
                                                         cond_dim_txts,
                                                         cond_dim_clusters,
                                                         rnn_dim, vocab_dim, emb_dim)
        self.encoder             = Encoder(inpt_dim, channels, btlnk, cond_model=cond_model,
                                           normalizer=normalizer_enc,
                                           dropout_rate=dropout_encoder_resblock)
        self.noise_img           = tf.keras.layers.GaussianNoise(noise_img)
        if auxilary:
            self.auxilary_loss =  Auxilary_loss(color_cond_type,
                                                txt_cond_type,
                                                cluster_cond_type,
                                                color_dim=color_cond_dim,
                                                cluster_dim=cluster_cond_dim,
                                                txt_dim=txt_cond_dim)

        # decoding
        self.decoder= Decoder(inpt_dim, channels[::-1], rnn_dim,
                              vocab_dim, emb_dim, cond_dim_colors,
                              cond_dim_txts, cond_dim_clusters, dropout_conditionals,
                              noise_color, noise_txt, color_cond_dim,
                              normalizer=normalizer_dec, color_cond_type=color_cond_type,
                              txt_cond_type=txt_cond_type, cluster_cond_type=cluster_cond_type)

        self.relu = tf.keras.layers.ReLU()
        # optimizers
        self.optimizer_enc= tf.keras.optimizers.Adam(lr_enc, beta_1=beta1, beta_2=beta2)
        self.optimizer_dec= tf.keras.optimizers.Adam(lr_dec, beta_1=beta1, beta_2=beta2)

    def encode(self, x, color=None, txt=None, cluster=None, training=False):
        cond = self.conditional_embedder(color, txt, cluster, training=training)
        z, mu, lv = self.encoder(x, cond, training=training)
        return z, mu, lv

    def decode(self, z, color=None, txt=None, cluster=None, training=False):
        cond = self.conditional_embedder(color, txt, cluster, training=training)
        return self.decoder(z, cond, training=training)

    def kl_loss(self, mu, lv):
        return tf.reduce_mean(-0.5 * tf.reduce_sum((-1*(tf.square(-mu)+tf.exp(lv))+1+lv),-1))+0.0

    def mse_loss(self, x, x_rec, size_average=True):
        x = tf.reduce_sum(tf.math.square(tf.reshape(x_rec-x,(x.shape[0],-1))), -1)
        if size_average:
            return tf.reduce_mean(x)+0.0
        else:
            return tf.reduce_sum(x)+0.0

    def vae_step(self, x, colors, txts, training=True):
        x = self.noise_img(self.inpt_layer(x), training=training)
        x_txt = self.inpt_layer_txt(txts) if self.txt_cond_type else None
        x_color = self.inpt_layer_cond(colors) if self.color_cond_type else None

        # encode
        z, mu, lv = self.encode(x, training=training)  # encode real image

        # decode
        x_rec = self.decode(z, x_color, x_txt, training=training)  # reconstruct real image

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

    @tf.function(experimental_relax_shapes=True)
    def train_vae(self, x, colors, txts):
        with tf.GradientTape() as tape:
            output = self.vae_step(x, colors, txts, training=True)
            loss = output["loss"]
        self.optimizer_enc.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return output


    def call(self, x, colors, txts, clusters, training=False):
        # inpts
        x         = self.inpt_layer(x)
        z_p       = tf.random.normal((self.batch_size, self.btlnk), 0, 1)

        x_txt     = self.inpt_layer_txt(txts) if self.txt_cond_type else None
        x_color   = self.inpt_layer_color(colors) if self.color_cond_type else None
        x_cluster = self.inpt_layer_cluster(clusters) if self.cluster_cond_type else None


        #########

        z, mu, lv = self.encode(x, x_color, x_txt, x_cluster, training=training)  # encode real image
        x_r       = self.decode(z, x_color, x_txt, x_cluster, training=training)  # reconstruct real image
        x_p       = self.decode(z_p, x_color, x_txt, x_cluster, training=training) # generate fake from z_p

        # reconstruction loss
        loss_rec =  self.mse_loss(x, x_r)* self.weight_rec

        # no gradient flow for encoder
        _, mu_r_, lv_r_ = self.encode(tf.stop_gradient(x_r), x_color, x_txt, x_cluster, training=training) # encode reconstruction
        _, mu_p_, lv_p_ = self.encode(tf.stop_gradient(x_p), x_color, x_txt, x_cluster, training=training) # encode fake

        # Encoder Adversarial Loss
        kl_real  = self.kl_loss(mu, lv)
        kl_rec_  = self.kl_loss(mu_r_, lv_r_)
        kl_fake_ = self.kl_loss(mu_p_, lv_p_)
        loss_enc_adv =(kl_real + 0.5*(self.relu(self.m_plus - kl_rec_) + self.relu(self.m_plus-kl_fake_))*self.weight_neg)* self.weight_kl

        ### ENCODER LOSS
        loss_enc = loss_rec + loss_enc_adv


        # gradient flow for decoder
        _, dec_mu_r, dec_lv_r = self.encode(x_r, x_color, x_txt, x_cluster, training=training) # encode reconstruction
        _, dec_mu_p, dec_lv_p = self.encode(x_p, x_color, x_txt, x_cluster, training=training) # encode fake



        ### DECODER
        kl_rec       = self.kl_loss(dec_mu_r, dec_lv_r)
        kl_fake      = self.kl_loss(dec_mu_p, dec_lv_p)
        loss_dec_adv = 0.5*(kl_rec + kl_fake) * self.weight_kl
        loss_dec     = loss_dec_adv + loss_rec

        if self.auxilary:
            # TODO input mu_r_??
            loss_aux, pred_cl_r, pred_cl_p, pred_color_r, pred_color_p = self.auxilary_loss(
                dec_mu_p, dec_mu_r,
                color_label=colors,
                cluster_label=clusters,
                txt_label=txts,
                training=training)
            loss_aux = self.weight_aux * loss_aux
            loss_enc += loss_aux
            loss_dec += loss_aux
        else:
            loss_aux = 0

        return {"x"            : x,
                "z_p"          : z_p,
                "x_r"          : x_r,
                "x_p"          : x_p,
                "loss_enc"     : loss_enc,
                "loss_dec"     : loss_dec,
                "loss_rec"     : loss_rec,
                "kl_real"      : kl_real,
                "kl_rec"       : kl_rec,
                "kl_fake"      : kl_fake,
                "mu"           : tf.reduce_mean(mu),
                "lv"           : tf.reduce_mean(lv),
                "loss_enc_adv" : loss_enc_adv,
                "loss_dec_adv" : loss_dec_adv,
                "loss_aux"     : loss_aux,
                "pred_cl_r"    : pred_cl_r,
                "pred_cl_p"    : pred_cl_p,
                "pred_color_r" : pred_color_r,
                "pred_color_p" : pred_color_p}


    @tf.function(experimental_relax_shapes=True)
    def train(self, x, colors, txts, clusters):
        with tf.GradientTape() as e_tape, tf.GradientTape() as d_tape:
            output = self.call(x, colors, txts, clusters, training=True)
            e_loss = output["loss_enc"] # encoder
            d_loss = output["loss_dec"] # decoder

        enc_variables = self.encoder.trainable_variables
        if self.cond_model=="enc_dec":
            enc_variables += self.conditional_embedder.trainable_variables
        if self.auxilary:
            enc_variables += self.auxilary_loss.trainable_variables
        self.optimizer_enc.apply_gradients(
            zip(e_tape.gradient(e_loss, enc_variables),
                enc_variables))

        dec_variables = self.decoder.trainable_variables + self.conditional_embedder.trainable_variables
        if self.auxilary:
            dec_variables += self.auxilary_loss.trainable_variables
        self.optimizer_dec.apply_gradients(
            zip(d_tape.gradient(d_loss, dec_variables),dec_variables))
        return output

class Auxilary_loss(tf.keras.layers.Layer):
    def __init__(self,
                 color_cond_type,
                 txt_cond_type,
                 cluster_cond_type,
                 color_dim=None,
                 cluster_dim=None,
                 txt_dim=None,
                 name="Auxilary_loss",
                 **kwargs):
        super(Auxilary_loss, self).__init__(name=name, **kwargs)
        self.color_cond_type = color_cond_type
        self.txt_cond_type = txt_cond_type
        self.cluster_cond_type = cluster_cond_type
        self.softmax = tf.keras.layers.Softmax()

        if color_cond_type:
            self.dense_color   = tf.keras.layers.Dense(color_dim, name="dense_color")
        if cluster_cond_type:
            self.dense_cluster = tf.keras.layers.Dense(cluster_dim, name="dense_cluster")
        if txt_cond_type:
            self.dense_txt     = tf.keras.layers.Dense(txt_dim, name="dense_txt")
    def call(self, z_p, z_r, color_label=None, cluster_label=None, txt_label=None, training=False):
        loss_aux = 0
        mean = 0
        c_r_cluster = None
        c_p_cluster = None
        c_r_color = None
        c_p_color = None
        if self.color_cond_type=="one_hot":
            c_r_color   = self.dense_color(z_r)
            c_p_color   = self.dense_color(z_p)
            aux_r_color = tf.keras.losses.categorical_crossentropy(color_label, c_r_color, from_logits=True)
            aux_p_color = tf.keras.losses.categorical_crossentropy(color_label, c_p_color, from_logits=True)
            loss_aux    += (aux_r_color + aux_p_color)/2
            mean        +=1
        #if self.color_cond_type=="continuous":
        #    c_r = self.dense_aux_cont(dec_mu_r)
        #    c_p = self.dense_aux_cont(dec_mu_p)
        #if self.txt_cond_type:
        #    c_r = self.dense_aux_txt(dec_mu_r)
        #    c_p = self.dense_aux_txt(dec_mu_p)
        c_r_cluster = None
        c_p_cluster = None
        if self.cluster_cond_type:
            c_r_cluster   = self.softmax(self.dense_cluster(z_r))
            c_p_cluster   = self.softmax(self.dense_cluster(z_p))
            aux_r_cluster = tf.keras.losses.categorical_crossentropy(cluster_label, c_r_cluster, from_logits=False)
            aux_p_cluster = tf.keras.losses.categorical_crossentropy(cluster_label, c_p_cluster, from_logits=False)
            loss_aux      += (aux_r_cluster + aux_p_cluster)/2
            mean          += 1
        loss_aux= tf.reduce_mean(loss_aux)/mean
        return loss_aux, c_r_cluster, c_p_cluster, c_r_color, c_p_color


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 inpt_dim,
                 channels,
                 latent_dim,
                 cond_model=None,
                 normalizer=None,
                 name="Encoder",
                 dropout_rate=0,
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.relu = tf.keras.layers.ReLU()

        self.cond_model = cond_model
        self.resize_shape      = (-1, inpt_dim[0], inpt_dim[1], 1)
        self.dense_resize = tf.keras.layers.Dense(inpt_dim[0]*inpt_dim[1],
                                                  name="dense_resize")

        self.layers = []
        self.layers.append(tf.keras.layers.Conv2D(channels[0],
                                         kernel_size=5,
                                         strides=1,
                                         padding="same",
                                         use_bias=False),)
        if normalizer:
            self.layers.append(normalizer())
        self.layers.append(tf.keras.layers.LeakyReLU(0.2))
        self.layers.append(tf.keras.layers.AveragePooling2D())

        for i, channel in enumerate(channels[1:], 1):
            self.layers.append(ResBlock(channel,
                                        adjust_channels=channel!=channels[i-1],
                                        normalizer=normalizer,
                                        dropout_rate=dropout_rate))
            self.layers.append(tf.keras.layers.AveragePooling2D())

        # additional res layer
        self.layers.append(ResBlock(channels[-1],
                                    normalizer=normalizer,
                                    dropout_rate=dropout_rate))

        self.variational = VariationalEncoding(latent_dim)

    def call(self, x, conditional=None, training=True):
        if self.cond_model=="enc_dec":
            cond = self.relu(self.dense_resize(conditional))
            cond = tf.reshape(cond, self.resize_shape)
            x = tf.concat([x, cond], -1)
        for layer in self.layers:
            x = layer(x, training=training)
        z, mu, lv = self.variational(x, training=training)
        return z, mu, lv


class Conditional_Embedder(tf.keras.layers.Layer):
    def __init__(self,
                 color_cond_type,
                 txt_cond_type,
                 cluster_cond_type,
                 cond_dim_colors,
                 cond_dim_txts,
                 cond_dim_clusters,
                 rnn_dim, vocab_dim, emb_dim,
                 name="Conditional_Embedder",
                 **kwargs):
        super(Conditional_Embedder, self).__init__(name=name, **kwargs)

        self.color_cond_type   = color_cond_type
        self.txt_cond_type     = txt_cond_type
        self.cluster_cond_type = cluster_cond_type
        self.relu              = tf.keras.layers.ReLU()
        if color_cond_type:
            self.dense_cond_color = tf.keras.layers.Dense(cond_dim_colors,
                                                          name="dense_cond_color")

        if txt_cond_type:
            self.dense_cond_txt = tf.keras.layers.Dense(cond_dim_txts,
                                                        name="dense_cond_txt")

            if txt_cond_type=="rnn":
                self.RNN = GRU_bidirectional(rnn_dim, vocab_dim, emb_dim)

        if cluster_cond_type:
            self.dense_cond_cluster = tf.keras.layers.Dense(cond_dim_clusters,
                                                            name="dense_cond_cluster")

    def call(self, color=None, txt=None, cluster=None, training=False):
        # if continous embedding or one hot then this layer is added
        cond_color = self.relu(self.dense_cond_color(color)) if self.color_cond_type else None
        if self.txt_cond_type=="rnn":
            txt = self.RNN(txt)
            #txt = self.relu(self.dense_cond_txt2(txt))
        cond_txt  = self.relu(self.dense_cond_txt(txt)) if self.txt_cond_type else None
        cond_cluster = self.relu(self.dense_cond_cluster(cluster)) if self.cluster_cond_type else None

        if self.cluster_cond_type and self.txt_cond_type and self.color_cond_type:
            return  tf.concat([cond_color, cond_txt, cond_cluster], 1)
        elif self.txt_cond_type and self.color_cond_type:
            return  tf.concat([cond_color, cond_txt], 1)
        elif self.txt_cond_type and self.cluster_cond_type:
            return  tf.concat([cond_cluster, cond_txt], 1)
        elif self.cluster_cond_type and self.color_cond_type:
            return  tf.concat([cond_color, cond_cluster], 1)
        elif self.txt_cond_type:
            return  cond_txt
        elif self.color_cond_type:
            return  cond_color
        elif self.cluster_cond_type:
            return  cond_cluster

class Decoder(tf.keras.layers.Layer):
    def __init__(self, inpt_dim,
                 channels,
                 rnn_dim,
                 vocab_dim,
                 emb_dim,
                 cond_dim_colors,
                 cond_dim_txts,
                 cond_dim_clusters,
                 dropout_conditionals,
                 noise_color,
                 noise_txt,
                 color_onehot_dim,
                 color_cond_type=None,
                 txt_cond_type=None,
                 cluster_cond_type=None,
                 normalizer=None,
                 name="Decoder",
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.relu = tf.keras.layers.ReLU()
        self.color_cond_type=color_cond_type
        self.txt_cond_type=txt_cond_type
        self.cluster_cond_type=cluster_cond_type

        xy_dim = inpt_dim[1]/(2**len(channels)) #wh
        self.dec_reshape_dim = (-1, int(xy_dim), int(xy_dim), channels[0])

        self.dense_resize     = tf.keras.layers.Dense(xy_dim*xy_dim*channels[0],
                                                      name="dense_resize")
        self.layers = []
        self.layers.append(ResBlock(channels[0],
                           normalizer=normalizer))
        self.layers.append(tf.keras.layers.UpSampling2D(size=(2,2)))

        for i,channel in enumerate(channels[1:], 1):
            self.layers.append(ResBlock(channel,
                                        adjust_channels= channel!=channels[i-1],
                                        normalizer=normalizer))
            self.layers.append(tf.keras.layers.UpSampling2D(size=(2,2)))

        # additional res layer
        self.layers.append(ResBlock(channels[-1],
                                    normalizer=normalizer))

        self.layers.append(tf.keras.layers.Conv2D(inpt_dim[-1], kernel_size=5,
                                                  strides=1, padding="same"))


    def call(self, z, conditional=None, training=False):
        if self.color_cond_type or self.txt_cond_type or self.cluster_cond_type:
            z = tf.concat([z, conditional], 1)

        x = self.relu(self.dense_resize(z))
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
