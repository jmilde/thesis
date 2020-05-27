from src.util_tf import batch_resize, batch, pipe, spread_image
from src.util_io import pform
from src.models.vae_gan import VAEGAN
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


    epochs     = 50
    batch_size = 32
    #warmup = ds_size//batch_size//2 # no latent loss for the first half epoch
    accelerate = (ds_size//batch_size)*1.5 # after warmup takes ~3 epochs until latent loss is considered 100%
    logfrq = ds_size//100//batch_size # log ~100x epoch
    loss_latent_scaling = 1
    loss_rec_scaling = 1

    # encoding
    btlnk = 800
    channels = [64, 128, 256, 512, 512]



    #ds_size = len(np.load(path_data)['imgs'])

    model_name = f"no_gan_vaeGan-{RESIZE_SIZE}-e{epochs}-b{batch_size}-btlnk{btlnk}-{channels}"
    path_ckpt  = path_ckpt+model_name
    #pipeline

    bg = batch_resize(path_data, batch_size, RESIZE_SIZE)
    data = pipe(lambda: bg, (tf.float32),prefetch=6)

    # model
    #inpt = tf.keras.Input(shape=img_dim)
    model = VAEGAN(img_dim,
                   channels,
                   btlnk,
                   batch_size,
                   accelerate = accelerate,
                   normalizer_enc = tf.keras.layers.BatchNormalization,
                   normalizer_dec = tf.keras.layers.BatchNormalization,
                   loss_latent_scaling = loss_latent_scaling,
                   loss_rec_scaling    = loss_rec_scaling,)
    #model = tf.keras.models.Model(inpt, architecture(inpt))


    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               #optimizer_enc_gen=model.optimizer_enc_gen,
                               #optimizer_dec=model.optimizer_dec,
                               net=model)
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
                    tf.summary.image( "x",
                                      spread_image(output["x"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.image( "x_rec",
                                      spread_image(output["x_rec"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.image( "x_noise",
                                      spread_image(output["x_noise"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.scalar("d_loss"        , output["d_loss"].numpy()        , step=step)
                    tf.summary.scalar("g_loss"        , output["g_loss"].numpy()        , step=step)
                    tf.summary.scalar("e_loss"        , output["e_loss"].numpy()        , step=step)
                    tf.summary.scalar("gx_rec_loss"   , output["gx_rec_loss"].numpy()   , step=step)
                    tf.summary.scalar("gx_noise_loss" , output["gx_noise_loss"].numpy() , step=step)
                    tf.summary.scalar("dx_loss"       , output["dx_loss"].numpy()       , step=step)
                    tf.summary.scalar("dx_rec_loss"   , output["dx_rec_loss"].numpy()   , step=step)
                    tf.summary.scalar("dx_noise_loss" , output["dx_noise_loss"].numpy() , step=step)
                    tf.summary.scalar("loss_rec"      , output["loss_rec"].numpy()      , step=step)
                    tf.summary.scalar("loss_latent"   , output["loss_latent"].numpy()   , step=step)
                    tf.summary.scalar("lv"            , output["lv"].numpy()            , step=step)
                    tf.summary.scalar("mu"            , output["mu"].numpy()            , step=step)
                    tf.summary.scalar("rate_anneal"   , output["rate_anneal"].numpy()   , step=step)
                    tf.summary.scalar("lr_balancer"   , output["lr_balancer"].numpy()   , step=step)
                    writer.flush()

        # save model every epoch
        save_path = manager.save()



if __name__=="__main__":
    main()
