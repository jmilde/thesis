from src.util_tf import batch_resize, batch_resize_cond, batch, pipe, spread_image
from src.util_io import pform
from src.models.introvae import INTROVAE
from src.analyze_introvae import run_tests
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
    cond_hdim  = 256 #64 #512
    epochs     = 50
    batch_size = 16
    logfrq = ds_size//100//batch_size # log ~100x per epoch
    vae_epochs = 0 # pretrain only vae
    btlnk = 512
    channels = [32, 64, 128, 256, 512, 512]

    ### loss weights
    #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
    weight_rec = 0.2 #0.05
    weight_kl  = 1
    weight_neg = 0.5 #alpha 0.1-0.5
    m_plus     = 260 #120 #  should be selected according to the value of β, to balance advaserial loss
    lr_enc= 0.0001
    lr_dec= 0.0001
    beta1 = 0.9 #0.5
    model_name = f"Icond{cond_hdim}-pre{vae_epochs}-{','.join(str(x) for x in RESIZE_SIZE)}-m{m_plus}-lr{lr_enc}b{beta1}-w_rec{weight_rec}"

    path_ckpt  = path_ckpt+model_name

    #pipeline
    #bg = batch_resize(path_data, batch_size, RESIZE_SIZE)
    #data = pipe(lambda: bg, (tf.float32), prefetch=6)
    bg = batch_resize_cond(path_data, path_cond, batch_size, RESIZE_SIZE)
    data = pipe(lambda: bg, (tf.float32, tf.float32), (tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None])), prefetch=6)

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



    # workaround for memoryleak ?
    tf.keras.backend.clear_session()

    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)

    if restore_model:
        manager = tf.train.CheckpointManager(ckpt, path_ckpt, checkpoint_name=restore_model)
        ckpt.restore(manager.latest_checkpoint)
        print("\nmodel restored\n")



    if vae_epochs:
        manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name + "_VAEpretrain")
        step=0
        for _ in trange(vae_epochs, desc="epochs", position=0):
            for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
                step+=1
                if step==1:
                    with writer.as_default():
                        tf.summary.trace_export(name="introvae_vae", step=0, profiler_outdir=path_log)

                # train step
                output = model.train_vae(next(data))

                # logging
                if step%logfrq==0:
                    with writer.as_default():
                         tf.summary.image( "vae_x",
                                      spread_image(output["x"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                         tf.summary.image( "vae_x_rec",
                                      spread_image(output["x_rec"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                         tf.summary.scalar("vae_loss_rec" , output["loss_rec"].numpy() , step=step)
                         tf.summary.scalar("vae_loss_kl"  , output["loss_kl"].numpy()  , step=step)
        save_path = manager.save()
        print("\nsaved VAE-model\n")


    # manager for intravae training
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name)



    # training and logging
    step=0
    for epoch in trange(epochs, desc="epochs", position=0):
        for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
            ckpt.step.assign_add(1)
            step+=1 #using ckpt.step leads to memory leak
            output = model.train(next(data))

            # get graph
            if step==1 and not vae_epochs:
                with writer.as_default():
                    tf.summary.trace_export(name="introvae", step=0, profiler_outdir=path_log)
            # logging
            if step%logfrq==0:
                with writer.as_default():
                    tf.summary.image( "overview",
                                      spread_image(
                                          np.concatenate((output["x"].numpy()[:3],
                                                          output["x_r"].numpy()[:3],
                                                          output["x_p"].numpy()[:3]),
                                                    axis=0),
                                          3,3,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.image( "x",
                                      spread_image(output["x"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.image( "x_r",
                                      spread_image(output["x_r"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.image( "x_p",
                                      spread_image(output["x_p"].numpy()[:16],4,4,RESIZE_SIZE[0],RESIZE_SIZE[1]),
                                      step=step)
                    tf.summary.scalar("loss_enc" , output["loss_enc"].numpy() , step=step)
                    tf.summary.scalar("loss_dec" , output["loss_dec"].numpy() , step=step)
                    tf.summary.scalar("loss_rec" , output["loss_rec"].numpy() , step=step)
                    tf.summary.scalar("mu"       , output["mu"].numpy()       , step=step)
                    tf.summary.scalar("lv"       , output["lv"].numpy()       , step=step)
                    tf.summary.scalar("kl_real"       , output["kl_real"].numpy()       , step=step)
                    tf.summary.scalar("kl_fake"       , output["kl_fake"].numpy()       , step=step)
                    tf.summary.scalar("kl_rec"       , output["kl_rec"].numpy()       , step=step)
                    tf.summary.scalar("loss_enc_adv"  , output["loss_enc_adv"].numpy()  , step=step)
                    tf.summary.scalar("loss_dec_adv"  , output["loss_dec_adv"].numpy()  , step=step)
                    writer.flush()

        # save model every epoch
        save_path = manager.save()
        print(f"\nsaved model after epoch {epoch}\n")
        run_tests(model, writer, next(data)[1][:4], btlnk, batch_size)



if __name__=="__main__":
    main()
