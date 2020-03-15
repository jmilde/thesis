import tensorflow as tf
from src.util import Record


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


class Quantizer(tf.keras.layers.Layer):
    # TODO: write option to use exponential moving averages instead of auxilary loss
    def __init__(self, nr_emb, b_commitment=1, **kwargs):
        self.nr_emb = nr_emb
        self.b_commitment = b_commitment #beta
        super(Quantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.emb_mtrx = self.add_weight(
            initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
            shape=(input_shape[-1], self.nr_emb),
            name='emb_mtrx',
            trainable=True)
        super(Quantizer, self).build(input_shape)

    def call(self, inpt):
        x = tf.reshape(inpt, [-1, tf.shape(inpt)[-1]]) # flatten x=[b*w*h, d]
        dist = (tf.reduce_sum(x**2, 1, keepdims=True) -
                2 * tf.matmul(x, self.emb_mtrx) +
                tf.reduce_sum(self.emb_mtrx**2, 0, keepdims=True))

        idx = tf.argmax(-dist, 1)

        ### for training overview
        idx_oh = tf.one_hot(idx, self.nr_emb)
        avg_probs = tf.reduce_mean(idx_oh, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))) #  A low perplexity indicates the probability distribution is good at predicting the sample

        idx = tf.reshape(idx, tf.shape(inpt)[:-1])
        q = tf.nn.embedding_lookup(tf.transpose(self.emb_mtrx, [1, 0]), idx) #quantize

        #loss
        l_commitment = tf.reduce_mean((tf.stop_gradient(q) - inpt)**2) # encoder output close to codebook
        l_codebook = tf.reduce_mean((q - tf.stop_gradient(inpt))**2) # bring codebook closer to encoder output
        loss = self.b_commitment * l_commitment + l_codebook

        # straight through gradient flow
        q = inpt+ tf.stop_gradient(q - inpt)
        return q, loss, perplexity

    def compute_output_shape(self,input_shape):
        return input_shape


class QuantizerEMA(tf.keras.layers.Layer):
    """
    exponential moving averages to update the embedding vectors
    """
    def __init__(self, nr_emb, commitment_weight=1, decay=0.99, epsilon=1e-5, **kwargs):
        self.nr_emb = nr_emb
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.epsilon = epsilon
        super(QuantizerEMA, self).__init__(**kwargs)

    def build(self, input_shape):
        self.emb_mtrx = self.add_weight(
            initializer=tf.random_normal_initializer(),
            shape=(input_shape[-1], self.nr_emb),
            name='emb_mtrx',
            trainable=False)
        self.ema_cluster_size= self.add_weight(shape=[self.nr_emb],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
        self.ema_dw = self.add_weight(shape=(input_shape[-1], self.nr_emb),
                                      initializer=tf.keras.initializers.Constant(value=self.emb_mtrx.numpy()),
                                      trainable=False)
        super(QuantizerEMA, self).build(input_shape)

    def update_embeddings(self, inpt, encodings):
        self.ema_cluster_size.assign(
            tf.keras.backend.moving_average_update(
                self.ema_cluster_size, tf.reduce_sum(encodings, axis=0), self.decay))
        self.ema_dw.assign(
            tf.keras.backend.moving_average_update(
                self.ema_dw, tf.matmul(inpt, encodings, transpose_a=True), self.decay))

        n = tf.reduce_sum(self.ema_cluster_size)
        self.emb_mtrx.assign(
            self.ema_dw / tf.reshape(
                n * (self.ema_cluster_size + 1e-5) / (n + self.nr_emb * 1e-5),
                [1, -1]))

    def call(self, inpt, training=False):
        x = tf.reshape(inpt, [-1, tf.shape(inpt)[-1]])# flatten x=[b*w*h, d]
        dist = (tf.reduce_sum(x**2, 1, keepdims=True) -
                2 * tf.matmul(x, self.emb_mtrx) +
                tf.reduce_sum(self.emb_mtrx**2, 0, keepdims=True))
        idx = tf.argmax(-dist, 1)
        idx_oh = tf.one_hot(idx, self.nr_emb)

        if training: self.update_embeddings(x, idx_oh)

        idx = tf.reshape(idx, tf.shape(inpt)[:-1])
        q = tf.nn.embedding_lookup(tf.transpose(self.emb_mtrx, [1, 0]), idx)

        # loss
        l_commitment = tf.reduce_mean((tf.stop_gradient(q) - inpt)**2)
        loss = self.commitment_weight * l_commitment

        # straight through gradient flow
        q = inpt + tf.stop_gradient(q - inpt)

        # logging
        avg_probs = tf.reduce_mean(idx_oh, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs *
                                       tf.math.log(avg_probs + 1e-10)))
        return q, loss, perplexity



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


class ResBlock(Record):
    def __init__(self, nr_filters):
            self.relu = tf.keras.layers.ReLU()
            self.conv1 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")
            self.conv2 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")

    def __call__(self, inpt):
        with tf.name_scope("ResBlock") as scope:
            x = self.conv1(inpt)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x+inpt)
            return x
