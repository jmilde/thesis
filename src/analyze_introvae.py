from datetime import datetime
from matplotlib import pyplot as plt
from os.path import expanduser
from src.models.introvae import INTROVAE
from src.util_io import pform
from src.util_np import np, vpack
from src.util_sp import load_spm
from src.util_tf import batch_resize, batch_cond_spm, batch, pipe, spread_image
from tqdm import trange,tqdm
import h5py
import os
import tensorflow as tf
import tensorflow_addons as tfa
from src.hyperparameter import params
from src.fid import calculate_frechet_distance
from src.prep_fid import calc_and_save_reference
from skimage.io import imsave
import math


def calculate_scores(model, data, writer, path_fid, path_inception, model_name,
                     batch_size, normalize, fid_samples_nr):
    print("save 50.000 generated samples")
    path_fid_data = os.path.join(path_fid, model_name)
    if not os.path.isdir(path_fid_data):
        os.mkdir(path_fid_data)
    norm= 255 if normalize else 1
    sample_nr = 0
    imgs= []
    for _ in tqdm(range(math.ceil(fid_samples_nr//batch_size))):
        inpt = next(data)
        output = model.train(*inpt)
        imgs.extend(output["x_p"].numpy())
        for img in output["x_p"]:
            if sample_nr<=fid_samples_nr:
                imsave(os.path.join(path_fid_data, f"{sample_nr}.png"), np.clip(img*norm, 0, 255).astype("uint8"))
                sample_nr += 1

            else:
                break

    print("Calculate MS-SSIM Score")
    imgs = np.array(imgs)[:-1] if len(imgs)%2 else np.array(imgs)
    imgs = np.clip(imgs*norm, 0, 255)
    split_a = imgs[:len(imgs)//2]
    split_b = imgs[len(imgs)//2:]

    ms_ssim = tf.math.reduce_mean(tf.image.ssim_multiscale(split_a,
                                                           split_b,
                                                           255,
                                                           filter_size=8))
    print(f"MS_SSIM: {ms_ssim}")
    with writer.as_default():
        tf.summary.scalar("MS_SSIM_score" , ms_ssim , step=0)
        writer.flush()

    print("caluclate mean and var")
    calc_and_save_reference(path_fid_data,
                            os.path.join(path_fid, f"{model_name}.npz"),
                            inception_path=path_inception)

    print("calculate FID Score")
    mu1= np.load(os.path.join(path_fid, f"{model_name}.npz"), allow_pickle=True)["mu"]
    sigma1= np.load(os.path.join(path_fid, f"{model_name}.npz"), allow_pickle=True)["sigma"]
    mu2= np.load(os.path.join(path_fid, f"mu_var_dataset.npz"), allow_pickle=True)["mu"]
    sigma2= np.load(os.path.join(path_fid, f"mu_var_dataset.npz"), allow_pickle=True)["sigma"]
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID SCORE: {fid_score}")
    with writer.as_default():
        tf.summary.scalar("FID_score"  , fid_score , step=0)
        writer.flush()


def show_img(img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()


def move_through_latent(z_a, z_b, nr_steps):
    z_a, z_b = np.asarray(z_a), np.array(z_b)
    step = np.asarray((z_b-z_a)/nr_steps)
    return np.array([z_a + step*i for i in range(1, nr_steps+1)])

def run_tests(model, writer, img_embs, colors, txts, spm, btlnk, img_dim,
              batch_size=16, step=0):
        np.random.seed(27)

        x_gen, zs_gen, x_txt, x_color = [], [], [], []
        for img_emb, color, txt in zip(img_embs, colors, txts):

            # from random noise with color and txt from real examples
            x    = np.random.normal(0,1,(batch_size, btlnk))
            cond_color = None
            if model.color_cond_type:
                cond_color = np.repeat(color[np.newaxis, :], batch_size, axis=0)
            cond_txt = None
            if model.txt_cond_type:
                cond_txt = np.repeat(txt[np.newaxis, :], batch_size, axis=0)
            x_gen.extend(model.decode(x, cond_color, cond_txt))

            # latent space walk from real image to random point
            _, mu, _ = model.encode(img_emb[np.newaxis, :])
            zs = move_through_latent(mu[0], x[0], batch_size)
            zs_gen.extend(model.decode( zs, cond_color, cond_txt))

            # text exploration
            if model.txt_cond_type:
                _, x, _ = model.encode(np.repeat(img_emb[np.newaxis, :], batch_size, axis=0))
                cond_color= None
                if model.color_cond_type:
                    cond_color = np.repeat(color[np.newaxis, :], batch_size, axis=0)
                t = [spm.encode_as_ids(t) for t in ["firma 1", "hallo", "was geht ab",  "kaltes bier", "vogel", "bird", "pelikan", "imperceptron", "albatros coding", "tree leaves", "nice coffee", "german engineering", "abcdef ghij", "klmnopq", "rstu vwxyz", "0123456789"]]
                cond_txt   = vpack(t, (batch_size, max(map(len,t))), fill=1,  dtype="int64")
                x_txt.extend(model.decode(x, cond_color, cond_txt))

            if model.color_cond_type:
                # color exploration
                if model.color_cond_type=="one_hot":
                    cond_color = []
                    for i in range(12):
                        zeros = np.zeros(12)
                        zeros[i]=1
                        cond_color.append(zeros)
                    cond_color = np.array(cond_color)
                    color_batchsize = 12
                elif model.color_cond_type=="continuous":
                    color_batchsize=8
                    # blue, black, red, white/green, rainbow, türkis/red/white, black/gold/white
                    cond_color = np.array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.58, 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.01,
                                            0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.39],
                                           [0.93, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.03,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
                                           [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.74, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.01, 0.01, 0.  , 0.  , 0.  , 0.21],
                                           [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.05, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.09, 0.02, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.05, 0.  ,
                                            0.  , 0.  , 0.02, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.73],
                                           [0.  , 0.  , 0.  , 0.  , 0.  , 0.01, 0.02, 0.  , 0.  , 0.02, 0.  ,
                                            0.04, 0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.  , 0.  , 0.  , 0.  ,
                                            0.02, 0.  , 0.01, 0.  , 0.03, 0.  , 0.  , 0.  , 0.  , 0.  , 0.02,
                                            0.02, 0.04, 0.  , 0.01, 0.01, 0.01, 0.  , 0.  , 0.06, 0.01, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.  , 0.  , 0.02, 0.01, 0.  ,
                                            0.  , 0.04, 0.05, 0.01, 0.  , 0.  , 0.  , 0.01, 0.5 ],
                                           [0.03, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.01,
                                            0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  ,
                                            0.  , 0.  , 0.01, 0.38, 0.  , 0.  , 0.  , 0.  , 0.  , 0.05, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.49],
                                           [0.2 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.16, 0.  , 0.  , 0.  , 0.02, 0.01,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.01, 0.04, 0.  , 0.  , 0.01, 0.12, 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.19, 0.01, 0.  , 0.  , 0.01, 0.02, 0.21],
                                           [0.  , 0.  , 0.  , 0.  , 0.  , 0.1 , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.11, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.16, 0.01, 0.  ,
                                            0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,
                                            0.  , 0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.59],])

                _, x, _ = model.encode(np.repeat(img_emb[np.newaxis, :],  color_batchsize, axis=0))
                cond_txt = None
                if model.txt_cond_type:
                    cond_txt = np.repeat(txt[np.newaxis, :],  color_batchsize , axis=0)
                x_color.extend(model.decode(x, cond_color, cond_txt))

        with writer.as_default():
            tf.summary.image( "change_x",
                              spread_image(x_gen,1*len(colors),batch_size,img_dim[0],img_dim[1]),
                              step=step)
            tf.summary.image( "latent_walk",
                              spread_image(zs_gen,1*len(colors),batch_size,img_dim[0],img_dim[1]),
                              step=step)
            if model.txt_cond_type:
                tf.summary.image( "change_text",
                                  spread_image(x_txt,1*len(colors),batch_size,img_dim[0],img_dim[1]),
                                  step=step)
            if model.color_cond_type:
                tf.summary.image( "change_color",
                                  spread_image(x_color,1*len(colors),color_batchsize, img_dim[0],img_dim[1]),
                                  step=step)
            writer.flush()


def load_model():
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
    beta2 = p['beta2']
    noise_color = p['noise_color']
    noise_txt = p['noise_txt']
    noise_img = p['noise_img']
    ds_size = len(np.load(path_cond, allow_pickle=True)["colors"])
    color_cond_type = p['color_cond_type']
    txt_cond_type = p['txt_cond_type']
    color_cond_dim = len(np.load(path_cond, allow_pickle=True)["colors_old" if color_cond_type=="one_hot" else "colors"][1])


    if p["vae_epochs"] and p["epochs"]:
        modeltype = f"INTRO{p['epochs']}_pre{p['vae_epochs']}-m{m_plus}-b1{beta1}b2{beta2}-w_rec{weight_rec}-w_neg{weight_neg}"
    elif p["epochs"]:
        modeltype = f"INTRO{p['epochs']}-m{m_plus}-lr{lr_enc}b1{beta1}b2{beta2}-w_rec{weight_rec}-w_neg{weight_neg}-w_neg{weight_neg}"
    else:
        modeltype = f"VAE{p['vae_epochs']}-b1{beta1}b2{beta2}"
    txt_info   = f"txt:({txt_cond_type}-dense{cond_dim_txts}-rnn{rnn_dim}-emb{emb_dim})"  if color_cond_type else ""
    color_info = f"color:({color_cond_type}{cond_dim_color})" if color_cond_type else ""
    model_name = (f"{modeltype}-lr{lr_enc}-z{btlnk}"
                  f"{color_info}-"
                  f"{txt_info}-"
                  f"{','.join(str(x) for x in img_dim)}")


    model_name="VAE20-lr0.0002-z256--128,128,3"

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
                     beta2 = beta2,
                     noise_color =noise_color,
                     noise_txt =noise_txt,
                     noise_img =noise_img,)
    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)


    # restore_model:
    manager = tf.train.CheckpointManager(ckpt, path_ckpt, checkpoint_name=model_name,  max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    print("\nmodel restored\n")
    return model


def main():
    #
    model =  load_model()
    # load sentence piece model
    spm = load_spm(path_spm + ".model")
    spm.SetEncodeExtraOptions("bos:eos") # enable start(=2)/end(=1) symbols

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
    run_tests(model, writer, conds, spm, btlnk, batch_size)


if __name__=="__main__":
    main()
