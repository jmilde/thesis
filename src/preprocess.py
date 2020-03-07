from src.util_tf import batch, pipe
from src.util import Record
import numpy as np
import h5py
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from datetime import datetime
import os

def show_img(img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()

def train():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
    path_log = f"./tmp/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    batch_size = 2
    inpt_dim = 3
    n_resblock=2
    filters= 256
    kernel_size=4
    emb_dim = 64
    nr_emb = 512
    epochs = 2
    logfrq = 10 # how many step between logs

    ds_size = len(h5py.File(path_data, 'r')['data'])

    imgs = h5py.File(path_data, 'r')['data']
    #xx=[tuple(map(lambda x: round(x,1),colorsys.rgb_to_hsv(*y))) for x in np.rollaxis(imgs[2], 0,3) for y in x]
    conditionals = []
    for img in tqdm(imgs):
        xx=[tuple(round(yy,1) for yy in y/255) for x in np.rollaxis(img, 0,3) for y in x]
        conditionals.append(Counter(xx).most_common(10))

    #pipeline
    bg = batch(path_data, batch_size)
    data = pipe(lambda: bg, (tf.float32),prefetch=4)

    # model
    with tf.name_scope('input'):
        inpt = tf.keras.Input(shape=(400,400,3))
    architecture = vqvae(filters, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size)
    model = tf.keras.models.Model(inpt, architecture(inpt))
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_model(x):
        return model(x, training=True)

    #logging
    writer = tf.summary.create_file_writer(path_log)
    tf.summary.trace_on(graph=True, profiler=True)


    # training and logging
    step = 0
    for _ in trange(epochs, desc="epochs", position=0):
        for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
            step += 1
            with  tf.GradientTape() as tape:
                output = train_model(data)
                loss = output["loss"]
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

            # get graph
            if step==1:
                with writer.as_default():
                    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=path_log)
            # logging
            if step//logfrq==0:
                with writer.as_default():
                    tf.summary.image("original", output["inpt"].numpy(), step=step, max_outputs=2)
                    tf.summary.image("reconstruction", output["img"].numpy(), step=step, max_outputs=2)
                    tf.summary.scalar("loss", loss.numpy(), step=step)
                    writer.flush()

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
        with tf.name_scope("Quantizer") as scope:
            x = tf.reshape(inpt, [-1, tf.shape(inpt)[-1]]) # flatten x=[b*w*h, d]
            dist = (tf.reduce_sum(x**2, 1, keepdims=True) -
                    2 * tf.matmul(x, self.emb_mtrx) +
                    tf.reduce_sum(self.emb_mtrx**2, 0, keepdims=True))

            idx = tf.argmax(-dist, 1)

            ### for training overview
            #idx_oh = tf.one_hot(idx, self.nr_emb)
            #avg_probs = tf.reduce_mean(idx_oh, 0)
            #perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))) #  A low perplexity indicates the probability distribution is good at predicting the sample

            idx = tf.reshape(idx, tf.shape(inpt)[:-1])
            q = tf.nn.embedding_lookup(tf.transpose(self.emb_mtrx, [1, 0]), idx) #quantize

            #loss
            l_commitment = tf.reduce_mean((tf.stop_gradient(q) - inpt)**2) # encoder output close to codebook
            l_codebook = tf.reduce_mean((q - tf.stop_gradient(inpt))**2) # bring codebook closer to encoder output
            loss = self.b_commitment * l_commitment + l_codebook

            # straight through gradient flow
            q = inpt+ tf.stop_gradient(q - inpt)

            return q, loss

    def compute_output_shape(
            self,
            input_shape
    ):
        return input_shape

class vqvae(Record):
    def __init__(self, filters, inpt_dim, emb_dim, nr_emb, n_resblock, kernel_size):
        n_resblock = n_resblock
        filters = filters
        kernel_size = kernel_size

        self.emb_dim = emb_dim
        self.nr_emb = nr_emb

        #encoder
        self.e_top = Encoder(n_resblock, filters, kernel_size, downsample=2)
        self.e_bot = Encoder(n_resblock, filters, kernel_size, downsample=4)

        # resizing for/after quantizer
        self.xt_rs = tf.keras.layers.Conv2D(self.emb_dim, kernel_size=1, padding="same")
        self.xb_rs = tf.keras.layers.Conv2D(self.emb_dim, kernel_size=1, padding="same")
        self.qxt_rs = tf.keras.layers.Conv2D(filters, kernel_size=1, padding="same")
        self.qxb_rs = tf.keras.layers.Conv2D(filters, kernel_size=1, padding="same")

        # quantizer
        self.q_top = Quantizer(emb_dim, nr_emb)
        self.q_bot = Quantizer(emb_dim, nr_emb)

        # decoder
        self.d_b = Decoder(inpt_dim, n_resblock, filters, kernel_size, upsample=4)
        self.d_t = Decoder(filters, n_resblock, filters, kernel_size, upsample=2)

    def __call__(self, inpt, latent_weight=1):
        with tf.name_scope("vqVAE-2") as scope:

            #TODO: check why others used permutations and concatinations for combining top/bottom
            # encode/downsample
            xb = self.e_bot(inpt) #[b, w/4, h/4, filters]
            xt= self.e_top(xb) #[b, w/8, h/8, filters]

            #### quantize
            ## top
            xt = self.xt_rs(xt) # kernel=1 conv to adjust filter number [b, w/8, h/8, emb_dim]
            qxt, l_t = self.q_top(xt) # quantization: [b, w/8, h/8, emb_dim]
            qxt = self.qxt_rs(qxt) # kernel=1 conv to adjust filter number: [b, w/8, h/8, filters]
            qxt = self.d_t(qxt) # [b, w/4, h/4, filters] # decode/upsample
            ## bot
            qxb, l_b = self.q_bot(self.xb_rs(xb + qxt)) # [b, w/4, h/4, emb_dim]
            qxb = self.qxb_rs(qxb) # kernel=1 conv to adjust filter number [b, w/4, h/4, filters]


            # decode/upsample
            img = tf.clip_by_value(self.d_b(qxb + qxt),0,1)

            loss_latent = tf.reduce_mean(l_t) + tf.reduce_mean(l_b)
            loss_mse =  tf.reduce_mean(tf.square(inpt-img))#reconstruction loss
            loss = latent_weight*loss_latent + loss_mse

            return {"inpt":inpt,
                    "img":img,
                    "loss": loss,
                    "qxt":qxt,
                    "qxb":qxb,}


class ResBlock(Record):
    def __init__(self, filters):
            self.relu = tf.keras.layers.ReLU()
            self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same")
            self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same")

    def __call__(self, inpt):
        with tf.name_scope("ResBlock") as scope:
            x = self.conv1(inpt)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x+inpt)
            return x


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
    def __init__(self, out_filter, n_resblock=2, filters=256, kernel_size=4, upsample=2):
        self.layers = [ResBlock(filters) for _ in range(n_resblock)]

        if upsample == 2:
            self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter,
                                                               kernel_size=4,
                                                               strides=2,
                                                               padding="same"))
        elif upsample == 4:
            self.layers.append(tf.keras.layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same"))
            self.layers.append(tf.keras.layers.ReLU())
            self.layers.append(tf.keras.layers.Conv2DTranspose(out_filter, kernel_size=4, strides=2, padding="same"))

    def __call__(self, x):
        with tf.name_scope("Decoder") as scope:
            for l in self.layers:
                x = l(x)
            return x
