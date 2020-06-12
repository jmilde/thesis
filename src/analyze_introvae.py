from src.util_tf import batch_resize, batch_resize_cond, batch, pipe, spread_image
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

def move_through_latent(z_a, z_b, nr_steps):
    z_a, z_b = np.asarray(z_a), np.array(z_b)
    step = np.asarray((z_b-z_a)/nr_steps)
    return [z_a + step*i for i in range(1, nr_steps+1)]

def run_tests(model, writer, conds, btlnk, batch_size=16, step=0):
        np.random.seed(27)

        x_gen, zs_gen = [], []
        for cond in conds:
            x    = np.random.rand(batch_size, btlnk)
            cond_batch = np.repeat(cond[np.newaxis, :], batch_size, axis=0)
            x_gen.extend(model.decode(x, cond_batch))

            zs = move_through_latent(x[0], x[1], batch_size)
            zs_gen.extend(model.decode(zs, cond_batch))

        with writer.as_default():
            tf.summary.image( "x_gen",
                              spread_image(x_gen,4,4*len(conds),256,256),
                              step=step)
            tf.summary.image( "zs_gen",
                              spread_image(zs_gen,1*len(conds),16,256,256),
                              step=0)
            writer.flush()


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    path_data = expanduser('~/data/LLD-logo.hdf5')
    path_cond = expanduser('~/data/color_conditional.npz')
    path_log  = expanduser("~/cache/tensorboard-logdir/")
    path_ckpt = expanduser('./ckpt/')

    # restore pretrained?
    restore_model = "" #empty or modelname for model stored at path_ckpt

    # Data info
    RESIZE_SIZE = [256,256]
    ds_size = len(h5py.File(path_data, 'r')['data'])
    INPUT_CHANNELS = 3
    img_dim = RESIZE_SIZE + [INPUT_CHANNELS]
    cond_dim = len(np.load(path_cond, allow_pickle=True)["colors"][1])
    cond_hdim  = 64 #64 #512
    epochs     = 50
    batch_size = 16
    logfrq = ds_size//100//batch_size # log ~100x per epoch
    vae_epochs = 0 # pretrain only vae
    btlnk = 512
    channels = [32, 64, 128, 256, 512, 512]

    ### loss weights
    #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
    weight_rec = 0.05 #0.5
    weight_kl  = 1
    weight_neg = 0.5 #alpha 0.1-0.5
    m_plus     = 120 #550  should be selected according to the value of β, to balance advaserial loss
    lr_enc= 0.0001
    lr_dec= 0.0001
    beta1 = 0.9
    model_name = f"Icond{cond_hdim}-pre{vae_epochs}-{','.join(str(x) for x in RESIZE_SIZE)}-m{m_plus}-lr{lr_enc}" #b{beta1}-w_rec{weight_rec}"
    path_ckpt  = path_ckpt+model_name

    # model
    model = INTROVAE(img_dim,
                     cond_dim,
                     channels,
                     btlnk,
                     batch_size,
                     cond_hdim,
                     normalizer_enc = tf.keras.layers.BatchNormalization,
                     normalizer_dec = tf.keras.layers.BatchNormalization,
                     weight_rec=weight_rec,
                     weight_kl=weight_kl,
                     weight_neg = weight_neg,
                     m_plus = m_plus,
                     lr_enc= lr_enc,
                     lr_dec= lr_dec,
                     beta1 = beta1)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)
    # logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))

    # restore_model:
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, checkpoint_name=restore_model,  max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    print("\nmodel restored\n")

    conds = np.array([[0.  , 0.6  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
               0.  , 0.  , 0.  , 0.4  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0. ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0., 0. ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0. ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0. ],
            [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.66,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.04, 0.  ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
               0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.3 ]])
    run_tests(model, writer, conds, btlnk, batch_size)


if __name__=="__main__":
    main()
