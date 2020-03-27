from src.util_tf import batch_resize, batch, pipe
from src.util_io import pform
from src.models.vae import VAE
import numpy as np
import h5py
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import os
from os.path import expanduser

def show_img(img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()

def main():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    path_data = expanduser('~/data/LLD-logo.hdf5')
    path_log  = expanduser("~/cache/tensorboard-logdir/")
    path_ckpt = expanduser('./ckpt/')
    #path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
    #path_log = f"./tmp/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Data info
    RESIZE_SIZE = [128,128]
    ds_size = len(h5py.File(path_data, 'r')['data'])
    INPUT_CHANNELS = 3
    img_dim = RESIZE_SIZE + [INPUT_CHANNELS]


    epochs = 50
    batch_size = 64
    #warmup = ds_size//batch_size//2 # no latent loss for the first half epoch
    accelerate = (ds_size//batch_size)*1.5 # after warmup takes ~3 epochs until latent loss is considered 100%
    logfrq = ds_size//100//batch_size # log ~100x epoch


    btlnk = 800
    channels = [64, 128, 256, 512, 512]



    #ds_size = len(np.load(path_data)['imgs'])

    model_name = f"vae_res-{RESIZE_SIZE}-e{epochs}-b{batch_size}-btlnk{btlnk}-{channels}"
    path_ckpt  = path_ckpt+model_name
    #pipeline

    bg = batch_resize(path_data, batch_size, RESIZE_SIZE)
    data = pipe(lambda: bg, (tf.float32),prefetch=6)

    # model
    #inpt = tf.keras.Input(shape=img_dim)
    model = VAE(img_dim,
                channels,
                btlnk,
                accelerate = accelerate,
                optimizer  = tf.keras.optimizers.Adam(),
                normalizer_enc = tf.keras.layers.BatchNormalization,
                normalizer_dec = tf.keras.layers.BatchNormalization,)
    #model = tf.keras.models.Model(inpt, architecture(inpt))



    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name)
    ckpt.restore(manager.latest_checkpoint)

    # workaround for memoryleak ?
    tf.keras.backend.clear_session()

    # training and logging
    step=0
    for _ in trange(epochs, desc="epochs", position=0):
        for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
            ckpt.step.assign_add(1)
            step+=1 #using ckpt.step leads to memory leak
            output = model.train(next(data), ckpt.step)


            # get graph
            if step==1:
                with writer.as_default():
                    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=path_log)
            # logging
            if step%logfrq==0:
                with writer.as_default():
                    tf.summary.image("original", output["x"].numpy(), step=step, max_outputs=2)
                    tf.summary.image("reconstruction", output["x_rec"].numpy(), step=step, max_outputs=2)
                    tf.summary.scalar("loss", output["loss"].numpy(), step=step)
                    tf.summary.scalar("loss_mse", output["loss_rec"].numpy(), step=step)
                    tf.summary.scalar("loss_latent", output["loss_latent"].numpy(), step=step)
                    tf.summary.scalar("lv", output["lv"].numpy(), step=step)
                    tf.summary.scalar("mu", output["mu"].numpy(), step=step)
                    tf.summary.scalar("rate_anneal", output["rate_anneal"].numpy(), step=step)
                    writer.flush()

        # save model every epoch
        save_path = manager.save()



if __name__=="__main__":
    main()
