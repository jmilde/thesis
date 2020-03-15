from src.util_tf import batch, pipe
from src.util_io import pform
from src.model import vqvae
import numpy as np
import h5py
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from datetime import datetime
import os
from os.path import expanduser

def show_img(img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()

def train():
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    #tf.config.experimental.set_virtual_device_configuration(
    #    gpus[0],
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])

    #os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #path_data = expanduser('~/data/LLD-logo.hdf5')
    #path_cond = expanduser('~/data/color_conditional.npz')
    #path_log  = expanduser("~/cache/tensorboard-logdir/")
    path_ckpt = expanduser('./ckpt/')
    path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
    path_cond = '/home/jan/Documents/uni/thesis/data/color_conditional.npz'
    path_log = f"./tmp/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    batch_size = 2
    inpt_channels, img_w, img_h = h5py.File(path_data, 'r')['data'][0].shape
    n_resblock=2
    filters= 256
    kernel_size=4

    emb_dim = 64
    nr_emb = 512

    epochs = 2
    logfrq = 100 # how many step between logs

    ds_size = len(h5py.File(path_data, 'r')['data'])
    cond_size = len(np.load(path_cond)['colors'][0])
    model_name = f"vqvae-b{batch_size}-res{n_resblock}-f{filters}-e{emb_dim}-nre{nr_emb}"

    #pipeline
    bg = batch(path_data, path_cond, batch_size)
    data = pipe(lambda: bg, (tf.float32, tf.float32),prefetch=4)

    # model
    inpt = tf.keras.Input(shape=(400,400,3))
    inpt = tf.keras.Input(shape=(cond_size))
    architecture = vqvae(filters, inpt_channels,img_w, img_h, emb_dim, nr_emb, n_resblock, kernel_size)
    model = tf.keras.models.Model(inpt, architecture(inpt, cond))
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_model(x):
        return model(x, training=True)

    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name)
    #status = checkpoint.restore(manager.latest_checkpoint)



    # training and logging
    step = 0
    for _ in trange(epochs, desc="epochs", position=0):
        for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
            step += 1
            ckpt.step.assign_add(1)
            with  tf.GradientTape() as tape:
                output = train_model(next(data))
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

        save_path = manager.save() # save every epoch


if __name__=="__main__":
    train()
