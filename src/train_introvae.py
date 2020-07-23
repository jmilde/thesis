from src.util_tf import batch_cond_spm, pipe, spread_image
from src.util_io import pform
from src.models.introvae import INTROVAE
from src.analyze_introvae import run_tests
from src.util_sp import load_spm
import numpy as np
import h5py
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import os
from os.path import expanduser
from src.hyperparameter import params

def show_img(
        img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    p = params["train"]

    path_ckpt = p['path_ckpt']
    path_cond = p['path_cond']
    path_data = p['path_data']
    path_log = p['path_log']
    path_spm = p['path_spm']
    restore_model = p['restore_model']
    img_dim = p['img_dim']
    btlnk = p['btlnk']
    channels = p['channels']
    cond_dim_color = p['cond_dim_color']
    rnn_dim = p['rnn_dim']
    cond_dim_txts = p['cond_dim_txts']
    emb_dim = p['emb_dim']
    dropout_conditionals = p['dropout_conditionals']
    dropout_encoder_resblock = p['dropout_encoder_resblock']
    vae_epochs = p['vae_epochs']
    epochs = p['epochs']
    batch_size = p['batch_size']
    logs_per_epoch = p['logs_per_epoch']
    normalizer_enc = p['normalizer_enc']
    normalizer_dec = p['normalizer_dec']
    weight_rec = p['weight_rec']
    weight_kl = p['weight_kl']
    weight_neg = p['weight_neg']
    m_plus = p['m_plus']
    lr_enc = p['lr_enc']
    lr_dec = p['lr_dec']
    beta1 = p['beta1']
    noise_color = p['noise_color']
    noise_txt = p['noise_txt']
    noise_img = p['noise_img']
    ds_size = len(np.load(path_cond, allow_pickle=True)["colors"])
    color_cond_type = p['color_cond_type']
    txt_cond_type = p['txt_cond_type']
    color_cond_dim = len(np.load(path_cond, allow_pickle=True)["colors_old" if color_cond_type=="one_hot" else "colors"][1])


    if p["vae_epochs"] and p["epochs"]:
        modeltype = f"INTRO{p['epochs']}_pre{p['vae_epochs']}-m{m_plus}-b{beta1}-w_rec{weight_rec}-w_neg{weight_neg}"
    elif p["epochs"]:
        modeltype = f"INTRO{p['epochs']}-m{m_plus}-lr{lr_enc}b{beta1}-w_rec{weight_rec}-w_neg{weight_neg}-w_neg{weight_neg}"
    else:
        modeltype = f"VAE{p['vae_epochs']}"
    txt_info   = f"txt:({txt_cond_type}-dense{cond_dim_txts}-rnn{rnn_dim}-emb{emb_dim})"  if color_cond_type else ""
    color_info = f"color:({color_cond_type}{cond_dim_color})" if color_cond_type else ""
    model_name = (f"{modeltype}-lr{lr_enc}-z{btlnk}"
                  f"{color_info}-"
                  f"{txt_info}-"
                  f"{','.join(str(x) for x in img_dim)}")

    logfrq = ds_size//logs_per_epoch//batch_size
    path_ckpt  = path_ckpt+model_name

    # load sentence piece model
    spm = load_spm(path_spm + ".model")
    spm.SetEncodeExtraOptions("bos:eos") # enable start(=2)/end(=1) symbols
    vocab_dim = spm.vocab_size()
    #pipeline
    #bg = batch_resize(path_data, batch_size, img_dim)
    #data = pipe(lambda: bg, (tf.float32), prefetch=6)
    bg = batch_cond_spm(path_data, path_cond, spm, batch_size, color_cond_type)
    data = pipe(lambda: bg, (tf.float32, tf.float32, tf.int64), (tf.TensorShape([None, None, None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None])), prefetch=6)


    # model
    model = INTROVAE(img_dim,
                     channels,
                     btlnk,
                     batch_size,
                     cond_dim_color,
                     rnn_dim,
                     cond_dim_txts,
                     vocab_dim,
                     emb_dim,
                     color_cond_dim,
                     color_cond_type,
                     txt_cond_type,
                     dropout_conditionals=dropout_conditionals,
                     dropout_encoder_resblock=dropout_encoder_resblock,
                     normalizer_enc = tf.keras.layers.BatchNormalization,
                     normalizer_dec = tf.keras.layers.BatchNormalization,
                     weight_rec=weight_rec,
                     weight_kl=weight_kl,
                     weight_neg = weight_neg,
                     m_plus = m_plus,
                     lr_enc= lr_enc,
                     lr_dec= lr_dec,
                     beta1 = beta1,
                     noise_color =noise_color,
                     noise_txt =noise_txt,
                     noise_img =noise_img,)

    # workaround for memoryleak ?
    tf.keras.backend.clear_session()

    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)

    if restore_model:
        manager = tf.train.CheckpointManager(ckpt, path_ckpt, checkpoint_name=model_name, max_to_keep=10)
        ckpt.restore(manager.latest_checkpoint)
        print("\nmodel restored\n")

    # for logging
    example_data=next(data)

    if vae_epochs:
        manager = tf.train.CheckpointManager(ckpt, path_ckpt, max_to_keep=3, checkpoint_name=model_name + "_VAEpretrain")
        step=0
        for _ in trange(vae_epochs, desc="epochs", position=0):
            for _ in trange(ds_size//batch_size, desc="steps in epochs", position=1, leave=False):
                step+=1
                if step==1:
                    with writer.as_default():
                        tf.summary.trace_export(name="introvae_vae", step=0, profiler_outdir=path_log)
                    run_tests(model, writer,example_data[0][:4], example_data[1][:4],
                              example_data[2][:4], spm, btlnk,
                              img_dim, batch_size=16, step=step,)


                # train step
                output = model.train_vae(*next(data))

                # logging
                if step%logfrq==0:
                    with writer.as_default():
                         tf.summary.image( "vae_x",
                                      spread_image(output["x"].numpy()[:16],4,4,img_dim[0],img_dim[1]),
                                      step=step)
                         tf.summary.image( "vae_x_rec",
                                      spread_image(output["x_rec"].numpy()[:16],4,4,img_dim[0],img_dim[1]),
                                      step=step)
                         tf.summary.scalar("vae_loss_rec" , output["loss_rec"].numpy() , step=step)
                         tf.summary.scalar("vae_loss_kl"  , output["loss_kl"].numpy()  , step=step)
                if step%(logfrq*10)==0:
                    run_tests(model, writer,example_data[0][:4], example_data[1][:4],
                              example_data[2][:4], spm, btlnk,
                              img_dim, batch_size=16, step=step,)

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
            output = model.train(*next(data))

            # get graph
            if step==1 and not vae_epochs:
                with writer.as_default():
                    tf.summary.trace_export(name="introvae", step=0, profiler_outdir=path_log)
                run_tests(model, writer,example_data[0][:4], example_data[1][:4],
                          example_data[2][:4], spm, btlnk,
                          img_dim, batch_size=16, step=step,)
            # logging
            if step%logfrq==0:
                with writer.as_default():
                    tf.summary.image( "overview",
                                      spread_image(
                                          np.concatenate((output["x"].numpy()[:3],
                                                          output["x_r"].numpy()[:3],
                                                          output["x_p"].numpy()[:3]),
                                                    axis=0),
                                          3,3,img_dim[0],img_dim[1]),
                                      step=step)
                    tf.summary.image( "x",
                                      spread_image(output["x"].numpy()[:16],4,4,img_dim[0],img_dim[1]),
                                      step=step)
                    tf.summary.image( "x_r",
                                      spread_image(output["x_r"].numpy()[:16],4,4,img_dim[0],img_dim[1]),
                                      step=step)
                    tf.summary.image( "x_p",
                                      spread_image(output["x_p"].numpy()[:16],4,4,img_dim[0],img_dim[1]),
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

            if step%(logfrq*10)==0:
                 run_tests(model, writer,example_data[0][:4], example_data[1][:4],
                              example_data[2][:4], spm, btlnk,
                              img_dim, batch_size=16, step=step,)

        # save model every epoch
        save_path = manager.save()
        print(f"\nsaved model after epoch {epoch}\n")



if __name__=="__main__":
    main()
