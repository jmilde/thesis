import numpy as np
import h5py
from tqdm import tqdm
import tensorflow as tf
from src.util_np import sample
from src.util import Record
from skimage.transform import resize

def spread_image(x, nrow, ncol, height, width):
    return tf.reshape(
        tf.transpose(
            tf.reshape(x, (nrow, ncol, height, width, -1))
            , (0, 2, 1, 3, 4))
        , (1, nrow * height, ncol * width, -1))

def batch(path, batch_size, seed=26, channel_first=False):
    """batch function to use with pipe"""
    ds = h5py.File(path, 'r')
    data = ds["data"]
    b = []
    for i in sample(len(data), seed):
        if batch_size == len(b):
            yield np.array(b, dtype=np.float32)
            b = []
        if channel_first:
            b.append(data[i].astype(np.float32)/255)
            #b.append(data[i].astype(np.float32)/255)
        else:
            b.append(resize(np.rollaxis(data[i], 0, 3), (384,384))/255)

def batch_resize(path, batch_size, size=(64,64), seed=26):
    """batch function to use with pipe"""
    ds     = h5py.File(path, 'r')
    data   = ds["data"]
    shapes = ds["shapes"]
    b = []
    for i in sample(len(data), seed):
        if batch_size == len(b):
            yield np.array(b, dtype=np.float32)
            b = []
        shape = shapes[i]
        b.append(resize(np.rollaxis(data[i][:,:shape[1], :shape[2]],0,3), size)/255)

def batch_resized(path, batch_size, seed=26, channel_first=False):
    """batch function to use with pipe"""
    data = np.load(path)["imgs"]
    b = []
    for i in sample(len(data), seed):
        if batch_size == len(b):
            yield np.array(b, dtype=np.float32)
            b = []
        if channel_first:
            b.append(np.rollaxis(data[i], 0, 3)/255)
        else:
            b.append(data[i]/255)

def pipe(generator, output_types, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    return tf.data.Dataset.from_generator(generator, output_types) \
                          .repeat(repeat) \
                          .prefetch(prefetch) \
                          .__iter__()


class downsampling_conv_block(Record):
    def __init__(self, filters, norm_func, strides=2, kernel_size=4):
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)
        self.norm = norm_func()
        self.relu = tf.keras.layers.ReLU()

    def __call__(self, inpt):
        with tf.name_scope("downsampling_block") as scope:
            x = self.conv(inpt)
            x = self.norm(x)
            x = self.relu(x)
        return x

class upsampling_conv_block(Record):
    def __init__(self, filters, norm_func, strides=2, kernel_size=4):
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=4,strides=2, padding="same")
        self.norm = norm_func()
        self.relu = tf.keras.layers.ReLU()

    def __call__(self, inpt):
        with tf.name_scope("upsampling_block") as scope:
            x = self.conv(inpt)
            x = self.norm(x)
            x = self.relu(x)
        return x


class downsampling_res_block(Record):
    def __init__(self, filters, normalizer, strides=2, kernel_size=3):
        self.relu = tf.keras.layers.ReLU()
        self.conv_down1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.normalize_down1 = normalizer()

        self.conv_down2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,strides=strides, padding="same")
        self.normalize_down2 = normalizer()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding="same")
        self.normalize = normalizer()


    def __call__(self, inpt):
        with tf.name_scope("upsampling_block") as scope:
            skip = self.conv_down1(inpt)
            #skip = self.normalize_down1(skip)

            x = self.conv_down2(inpt)
            # x = self.normalize_down2()
            x = self.relu(x)
            x = self.conv(x)
            #x = self.normalize(x)

            x = self.relu(x + skip)
        return x

class upsampling_res_block(Record):
    def __init__(self, filters, normalizer, strides=2, kernel_size=3):
        self.relu = tf.keras.layers.ReLU()
        self.conv_up1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.normalize_up1 = normalizer()

        self.conv_up2 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size,strides=strides, padding="same")
        self.normalize_up2 = normalizer()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding="same")
        self.normalize = normalizer()


    def __call__(self, inpt):
        with tf.name_scope("upsampling_block") as scope:
            skip = self.conv_up1(inpt)
            #skip = self.normalize_up1(skip)

            x = self.conv_up2(inpt)
            #x = self.normalize_up2(x)
            x = self.relu(x)
            x = self.conv(x)
            #x = self.normalize(x)

            x = self.relu(x + skip)
        return x

def sigmoid(x, shift=0.0, mult=20):
    """ squashes a value with a sigmoid"""
    return tf.constant(1.0) / (tf.constant(1.0) + tf.exp(-tf.constant(1.0) * (x * mult)))

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, nr_filters, adjust_channels=False, normalizer=tf.keras.layers.BatchNormalization, activation=tf.keras.layers.LeakyRelu(alpha=0.2)):
        super(ResBlock, self).__init__(name='ResBlock')


        if adjust_channels: self.adjust = tf.keras.layers.Conv2D(nr_filters, kernel_size=1, padding="same", use_bias=False)
        else: self.adjust = None

        self.activation = activation
        self.conv1 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same", use_bias=False)
        self.normalize1 = normalizer()
        self.conv2 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same", use_bias=False)
        self.normalize2 = normalizer()

    def call(self, inpt):
        with tf.name_scope("ResBlock") as scope:
            if self.adjust: inpt = self.adjust(inpt)
            x = self.conv1(inpt)
            x = self.normalize1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(self.normalize2(x+inpt))
        return x

#class ResBlock(Record):
#    def __init__(self, nr_filters, normalizer=tf.keras.layers.BatchNormalization):
#            self.relu = tf.keras.layers.ReLU()
#            self.conv1 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")
#            self.normalize1 = normalizer()
#            self.conv2 = tf.keras.layers.Conv2D(nr_filters, kernel_size=3, padding="same")
#            self.normalize2 = normalizer()
#
#    def __call__(self, inpt):
#        with tf.name_scope("ResBlock") as scope:
#            x = self.conv1(inpt)
#            #x = self.normalize1(x)
#            x = self.relu(x)
#            x = self.conv2(x)
#            #x = self.normalize2(x)
#            x = self.relu(x+inpt)
#        return x


class VariationalEncoding(tf.keras.layers.Layer):
    def __init__(self, btlnk,
                 name="VariationalEncoding",
                 **kwargs):
        super(VariationalEncoding, self).__init__(name=name, **kwargs)
        self.relu = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        #self.dense_btlnk = tf.keras.layers.Dense(btlnk)
        #self.normalize = tf.keras.layers.LayerNormalization()
        self.dense_mu = tf.keras.layers.Dense(btlnk)
        self.dense_lv = tf.keras.layers.Dense(btlnk)

    def call(self, x, training=True):
        with tf.name_scope("variational") as scope:
            x = self.flatten(x)
            #x = self.relu(self.dense_btlnk(x))
            #x = self.normalize()
            with tf.name_scope("latent") as scope:
                mu = self.dense_mu(x)
                lv = self.dense_lv(x)
            with tf.name_scope('z') as scope:
                z = mu + tf.exp(0.5 * lv) * tf.random.normal(shape=tf.shape(lv))
        return z, mu, lv


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
