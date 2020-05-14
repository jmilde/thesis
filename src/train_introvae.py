from src.util_tf import batch_resize, batch, pipe, spread_image
from src.util_io import pform
from src.models.introvae import INTROVAE
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

    # Data info
    RESIZE_SIZE = [128,128]
    ds_size = len(h5py.File(path_data, 'r')['data'])
    INPUT_CHANNELS = 3
    img_dim = RESIZE_SIZE + [INPUT_CHANNELS]


    epochs     = 50
    batch_size = 32
    logfrq = ds_size//100//batch_size # log ~100x epoch

    vae_epochs = 0 # pretrain only vae
    # loss weights
    weight_rec=1,
    weight_kl=1,
    weight_neg = 1
    m_plus = 100
    lr_enc= 0.0002,
    lr_dec= 0.0002,
    beta1 = 0.5




    # encoding
    btlnk = 512
    channels = [64, 128, 256, 512, 512]



    #ds_size = len(np.load(path_data)['imgs'])

    model_name = f"Introvae-{RESIZE_SIZE}-e{epochs}-b{batch_size}-btlnk{btlnk}-{channels}"
    path_ckpt  = path_ckpt+model_name
    #pipeline

    bg = batch_resize(path_data, batch_size, RESIZE_SIZE)
    data = pipe(lambda: bg, (tf.float32),prefetch=6)

    # model
    #inpt = tf.keras.Input(shape=img_dim)
    model = INTROVAE(img_dim,
                     channels,
                     btlnk,
                     batch_size,
                     normalizer_enc = tf.keras.layers.BatchNormalization,
                     normalizer_dec = tf.keras.layers.BatchNormalization,
                     weight_rec=weight_rec,
                     weight_kl=weight_kl,
                     weight_neg = weight_neg,
                     m_plus = m_plus,
                     lr_enc= lr_enc,
                     lr_dec= lr_dec,
                     beta1 = beta1)


    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name)
    ckpt.restore(manager.latest_checkpoint)

    # workaround for memoryleak ?
    tf.keras.backend.clear_session()


    if vae_epochs:
        step=0
        for _ in trange(vae_epochs, desc="epochs", position=0):
            for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
                output = model.train_vae(next(data))
                if step%logfrq==0:
                    with writer.as_default():
                         tf.summary.image( "vae_x",
                                      spread_image(output["x"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                         tf.summary.image( "vae_x_rec",
                                      spread_image(output["x_rec"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                         tf.summary.scalar("vae_loss_rec" , output["loss_rec"].numpy() , step=step)
                         tf.summary.scalar("vae_kl_real"  , output["kl_real"].numpy()  , step=step)
    # training and logging
    step=0
    for _ in trange(epochs, desc="epochs", position=0):
        for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
            ckpt.step.assign_add(1)
            step+=1 #using ckpt.step leads to memory leak
            output = model.train(next(data))

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
                    tf.summary.image( "x_fake",
                                      spread_image(output["x_fake"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.scalar("loss_enc" , output["loss_enc"].numpy() , step=step)
                    tf.summary.scalar("loss_dec" , output["loss_dec"].numpy() , step=step)
                    tf.summary.scalar("loss_rec" , output["loss_rec"].numpy() , step=step)
                    tf.summary.scalar("kl_real"  , output["kl_real"].numpy()  , step=step)
                    tf.summary.scalar("kl_fake"  , output["kl_fake"].numpy()  , step=step)
                    tf.summary.scalar("kl_rec"   , output["kl_rec"].numpy()   , step=step)
                    tf.summary.scalar("mu"       , output["mu"].numpy()       , step=step)
                    tf.summary.scalar("lv"       , output["lv"].numpy()       , step=step)
                    tf.summary.scalar("mu_rec"   , output["mu_rec"].numpy()   , step=step)
                    tf.summary.scalar("lv_rec"   , output["lv_rec"].numpy()   , step=step)
                    tf.summary.scalar("mu_fake"  , output["mu_fake"].numpy()  , step=step)
                    tf.summary.scalar("lv_fake"  , output["lv_fake"].numpy()  , step=step)
                    writer.flush()

        # save model every epoch
        save_path = manager.save()



if __name__=="__main__":
    main()
