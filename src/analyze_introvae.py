from skimage import io
from datetime import datetime
from collections import Counter
from matplotlib import pyplot as plt
from os.path import expanduser
from src.models.introvae import INTROVAE
from src.util_io import pform
from src.util_np import np, vpack, sample
from src.util_sp import load_spm
from src.util_tf import batch_cond_spm, pipe, spread_image
from tqdm import trange, tqdm
import h5py
import os
import tensorflow as tf
import tensorflow_addons as tfa
from src.hyperparameter import params
from src.fid import calculate_frechet_distance
from src.prep_fid import calc_and_save_reference
from src.prep_dataset import get_colors_old
from skimage.io import imsave
import math
import pandas as pd
from sklearn import metrics
from skimage.io import imsave
from scipy.stats import truncnorm

def generate_imgs(model, data, path_fid_data, model_name, fid_samples_nr, batch_size, training=False,
                  truncate=None, btlnk=256):
    bn = "bn" if training else ""
    modeltype = "intro" if "intro" in model_name.lower() else "vae"

    img_path = os.path.join(path_fid_data, f"imgs{bn}")
    print("save imgs to ", img_path)
    if not os.path.isdir(img_path):
        os.mkdir(img_path)

    sample_nr = 0
    imgs, colors, txts, clusters = [], [], [], []
    for _ in tqdm(range(math.ceil(fid_samples_nr//batch_size+1))):
        img, color, txt, cluster = next(data)
        if truncate:
            z = truncnorm.rvs(-truncate,truncate, size=(batch_size,btlnk))
        else:
            z = np.random.normal(0,1,(batch_size, btlnk))
        imgs_array = model.decode(z, color, txt, cluster, training=training).numpy()
        imgs.extend(imgs_array)
        for img in imgs_array:
            if sample_nr<=fid_samples_nr:
                imsave(os.path.join(img_path,
                                    f"{sample_nr}.png"),
                       np.clip(img*255, a_min=0, a_max=255).astype("uint8"))
                sample_nr += 1
                if model.color_cond_type:
                    colors.append(color)
                if model.txt_cond_type:
                    txts.append(txt)
                if model.cluster_cond_type:
                    clusters.append(cluster)
            else:
                break
    print("saving conditionals")
    try:
        np.savez_compressed(os.path.join(path_fid_data, f"conditionals{bn}.npz"),
                            colors= colors,
                            txts = txts,
                            clusters = clusters)
    except:
        print(colors, txts, clusters)
    return imgs

def evaluate_one_hot(model, data, writer):
    if model.color_cond_type == "one_hot":
        one_hot_dim = 11
    else:
        one_hot_dim = 10

    pred, labels, plot = [], [], []
    for i in tqdm(range(one_hot_dim)):
        for j in range(2):
            z = np.random.normal(0,1,(50, 256))
            zeros = np.zeros((50,one_hot_dim))
            zeros[:,i]=1
            labels.extend([i]*50)
            if model.color_cond_type=="one_hot":
                #for l in model.decode(z, zeros, None, None ):
                #    print(l)
                samples=  model.decode(z, zeros, None, None,training=False)
                if j==0:
                    plot.extend(samples[:10])

                imgs = [get_colors_old(l.numpy()*255) for l in samples]

                for l in imgs:
                    try:
                        pred.append(list(l).index(max(l)))
                    except:
                        pred.append(0)
                #pred.extend([list(l).index(max(l)) for l in imgs])
            else:
                samples=  model.decode(z, None, None, zeros, training=False)
                if j==0:
                    plot.extend(samples[:10])
                imgs = [get_colors_old(l.numpy()*255) for l in samples]
                for l in imgs:
                    try:
                        pred.append(list(l).index(max(l)))
                    except:
                        pred.append(0)


    precision = [round(x,2) for x in metrics.precision_score(labels, pred, average=None)]
    p_mean = round(np.mean(precision),2)
    print(f"precision: {precision}")
    print(f"mean: {p_mean}")
    f1 =[round(x,2) for x in metrics.f1_score(labels, pred, average=None)]
    f1_mean = round(np.mean(f1),2)
    print(f'F-Measure: {f1}')
    print(f"mean: {f1_mean}")
    recall = [round(x,2) for x in metrics.recall_score(labels, pred, average=None)]
    r_mean = round(np.mean(recall),2)
    print(f"recall: {recall}")
    print(f"mean: {r_mean}")

    colors = ['green','purple','black','brown','blue','cyan','yellow','gray','red', 'pink', 'orange']
    for i,(c,p,r,f) in enumerate(zip(colors, precision, recall, f1)):
        if model.color_cond_type == "one_hot":
            print(f"{c} & {p} & {r} & {f} \\\\")
        else:
            print(f"cluster{i+1} & {p} & {r} & {f} \\\\")
    print(f"mean & {p_mean} & {r_mean} & {f1_mean} \\\\")

    print("spread image")
    #i = spread_image(np.array(plot),10,11,128,128)
    print("save img")
    #imsave(expanduser("/home/users/jmilde/data/imgsplot.png"), np.clip(np.array(i[0])*255, a_min=0, a_max=255))

    print("write to tensorboard")
    with writer.as_default():
        tf.summary.image("one-hot-exploration", spread_image(np.array(plot),10,one_hot_dim,128,128), step=3)
        tf.summary.scalar("recall"  , round(r_mean,2) , step=0)
        tf.summary.scalar("f1"  , round(f1_mean,2) , step=0)
        tf.summary.scalar("precision"  , round(p_mean,2) , step=0)
        writer.flush()

    print("Done")




def mssim_dataset(path_imgs, path_cond, spm, batch_size, cond_type_color="old",
                   cond_type_txt="bert", cond_cluster_type="vgg",
                   txt_len_min=0, txt_len_max=9999, seed=26):
    """batch function to use with pipe
    cond_type_color = 'one_hot' or 'continuous'
    """
    no_txt = True if (("only" in path_imgs) or ("lld" in path_imgs)) else False

    color_cond = "colors_old" if cond_type_color=="one_hot" else "colors"
    txt_len = list(map(len,  np.load(path_cond, allow_pickle=True)["txts"]))
    relevant_idxs = [i for i,l in enumerate(txt_len) if (txt_len_min<=l<=txt_len_max) or no_txt]
    print("getting imgs")
    i = []
    for l in sample(len(relevant_idxs), seed):
        if len(i)==50000:
            break
        j = relevant_idxs[l]
        i.append(io.imread(os.path.join(path_imgs, f"{j}.png")))
    print("calculating mssim")
    ms_ssim=calc_mssim(np.array(i))
    print(f"mssim dataset: {ms_ssim}")


def calc_mssim(imgs):
    with tf.device('/CPU:0'):

        x = []
        for a,b in zip(np.array_split(imgs[:len(imgs)//2], 5),
                       np.array_split(imgs[len(imgs)//2:], 5)):
            x.extend( tf.image.ssim_multiscale(a,
                                               b,
                                               255,
                                               filter_size=8))
        ms_ssim = tf.math.reduce_mean(x)
    return ms_ssim

def fid_dataset():
    fid_path=expanduser("~/data/fid/")
    calc_and_save_reference(expanduser("~/data/lld_boosted"),
                            os.path.join(fid_path, "lld_fid_sample.npz"),
                            inception_path=path_inception,
                            nr_samples=50000)
    mu1= np.load(os.path.join(fid_path, f"lld_fid_sample.npz"), allow_pickle=True)["mu"]
    sigma1= np.load(os.path.join(fid_path, f"lld_fid_sample.npz"), allow_pickle=True)["sigma"]
    mu2= np.load(os.path.join(fid_path, path_fid_dataset), allow_pickle=True)["mu"]
    sigma2= np.load(os.path.join(fid_path, path_fid_dataset), allow_pickle=True)["sigma"]
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID SCORE: {fid_score}")

    fid_path=expanduser("~/data/fid/")
    calc_and_save_reference(expanduser("~/data/imgs"),
                            os.path.join(fid_path, "txt_fid_sample.npz"),
                            inception_path=path_inception,
                            nr_samples=50000)
    mu1= np.load(os.path.join(fid_path, f"txt_fid_sample.npz"), allow_pickle=True)["mu"]
    sigma1= np.load(os.path.join(fid_path, f"txt_fid_sample.npz"), allow_pickle=True)["sigma"]
    mu2= np.load(os.path.join(fid_path, path_fid_dataset), allow_pickle=True)["mu"]
    sigma2= np.load(os.path.join(fid_path, path_fid_dataset), allow_pickle=True)["sigma"]
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID SCORE: {fid_score}")



def calculate_scores(model, data, writer, path_fid, path_inception, model_name,
                     batch_size, fid_samples_nr, path_fid_dataset, plot_bn=False, truncate=None, btlnk=256):
    path_fid_data = os.path.join(path_fid, model_name)
    if not os.path.isdir(path_fid_data):
        os.mkdir(path_fid_data)
    if not truncate:
        truncate = ''

    #if model.color_cond_type=="one_hot" or model.cluster_cond_type=="vgg":
    #    print("one-hot exploration")
    #    evaluate_one_hot(model, data, writer)

    print("save 50.000 generated samples")
    imgs = generate_imgs(model, data, path_fid_data, model_name,
                         fid_samples_nr, batch_size, training=False, truncate=truncate,btlnk=btlnk)

    print("Calculate MS-SSIM Score")
    with tf.device('/CPU:0'):
        imgs = np.clip(np.array(imgs[:50000])*255, 0 , 255)
        ms_ssim = calc_mssim(imgs)
        print(f"MS_SSIM: {ms_ssim}")

        with writer.as_default():
            tf.summary.scalar(f"MS_SSIM_score{truncate}" , ms_ssim , step=0)
            writer.flush()#

        print("caluclate mean and var")
        bn = "bn" if plot_bn else ""
        calc_and_save_reference(os.path.join(path_fid_data, f"imgs{bn}"),
                                os.path.join(path_fid_data, f"{model_name}.npz"),
                                inception_path=path_inception)

        print("calculate FID Score")
        mu1= np.load(os.path.join(path_fid_data, f"{model_name}.npz"), allow_pickle=True)["mu"]
        sigma1= np.load(os.path.join(path_fid_data, f"{model_name}.npz"), allow_pickle=True)["sigma"]
        mu2= np.load(os.path.join(path_fid, path_fid_dataset), allow_pickle=True)["mu"]
        sigma2= np.load(os.path.join(path_fid, path_fid_dataset), allow_pickle=True)["sigma"]
        fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        print(f"FID SCORE: {fid_score}")
        with writer.as_default():
            tf.summary.scalar(f"FID_score{truncate}"  , fid_score , step=0)
            writer.flush()

    print("cleaning up")
    img_path = os.path.join(path_fid_data, f"imgs")
    for f in os.listdir(img_path):
        os.remove(os.path.join(img_path, f))


    if plot_bn:
        print("save 50.000 generated samples WITH BATCHNORM")
        imgs = generate_imgs(model, data, path_fid, model_name,
                             fid_samples_nr, batch_size, training=True)

        print("Calculate MS-SSIM Score")
        with tf.device('/CPU:0'):
            imgs = np.clip(np.array(imgs[:50000])*255, 0 , 255)
            x = []
            for a,b in zip(np.array_split(imgs[:len(imgs)//2], 5),
                           np.array_split(imgs[len(imgs)//2:], 5)):
                x.extend( tf.image.ssim_multiscale(a,
                                                   b,
                                                   255,
                                                   filter_size=8))
            ms_ssim = tf.math.reduce_mean(x)
            print(f"MS_SSIM BN: {ms_ssim}")

            with writer.as_default():
                tf.summary.scalar("MS_SSIM_score_BN" , ms_ssim , step=0)
                writer.flush()

            print("caluclate mean and var")
            calc_and_save_reference(path_fid_data,
                                os.path.join(path_fid, f"{model_name}_BN.npz"),
                                inception_path=path_inception)

            print("calculate FID Score")
            mu1= np.load(os.path.join(path_fid, f"{model_name}_BN.npz"), allow_pickle=True)["mu"]
            sigma1= np.load(os.path.join(path_fid, f"{model_name}_BN.npz"), allow_pickle=True)["sigma"]
            mu2= np.load(os.path.join(path_fid, path_fid_dataset), allow_pickle=True)["mu"]
            sigma2= np.load(os.path.join(path_fid, path_fid_dataset), allow_pickle=True)["sigma"]
            fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            print(f"FID SCORE BN: {fid_score}")
            with writer.as_default():
                tf.summary.scalar("FID_score_BN"  , fid_score , step=0)
                writer.flush()

        print("cleaning up")
        img_path = os.path.join(path_fid_data, f"imgsbn")
        for f in os.listdir(img_path):
            os.remove(os.path.join(img_path, f))#





def show_img(img, channel_first=False):
    if channel_first:
        img = np.rollaxis(img, 0,3)
    plt.imshow(img)
    plt.show()


def move_through_latent(z_a, z_b, nr_steps):
    z_a, z_b = np.asarray(z_a), np.array(z_b)
    step = np.asarray((z_b-z_a)/nr_steps)
    return np.array([z_a + step*i for i in range(1, nr_steps+1)])


def run_tests(model, writer, img_embs, colors, txts, clusters, spm, btlnk, img_dim,
              batch_size=16, step=0):
        np.random.seed(27)

        x_gen, zs_gen, x_txt, x_color, x_cluster= [], [], [], [], []
        for img_emb, color, txt, cluster in zip(img_embs, colors, txts, clusters):

            # from random noise with color and txt from real examples
            x    = np.random.normal(0,1,(batch_size, btlnk))
            cond_color = np.repeat(color[np.newaxis, :], batch_size, axis=0) if model.color_cond_type else None
            cond_txt = np.repeat(txt[np.newaxis, :], batch_size, axis=0)  if model.txt_cond_type else None
            cond_cluster = np.repeat(cluster[np.newaxis, :], batch_size, axis=0)  if model.cluster_cond_type else None
            x_gen.extend(model.decode(x, cond_color, cond_txt, cond_cluster).numpy())

            # latent space walk from real image to random point
            _, mu, _ = model.encode(img_emb[np.newaxis, :], color[np.newaxis, :], txt[np.newaxis, :], cluster[np.newaxis, :])
            zs = move_through_latent(mu[0], x[0], batch_size)
            zs_gen.extend(model.decode( zs, cond_color, cond_txt, cond_cluster))

            # text exploration
            if model.txt_cond_type:
                txt_samples = ["eggcellent", "easyBee", "aircoustic",  "zoneco", "vogel", "bird", "pelikan", "imperceptron", "albatros coding", "tree leaves", "nice coffee", "german engineering", "abcdef ghij", "klmnopq", "rstu vwxyz", "0123456789"]
                cond_color= None
                if model.color_cond_type:
                    cond_color = np.repeat(color[np.newaxis, :], batch_size, axis=0)
                if model.cluster_cond_type:
                    cond_cluster = np.repeat(cluster[np.newaxis, :], batch_size, axis=0)
                if model.txt_cond_type=="rnn":
                    t = [spm.encode_as_ids(t) for t in txt_samples]
                    cond_txt = vpack(t, (batch_size, max(map(len,t))), fill=1,  dtype="int64")
                elif model.txt_cond_type=="bert":
                    # used prep_bert.py to get the bert embeddingsfor the txt_samples
                    cond_txt = np.load(os.path.expanduser("~/data/txt.npz"),
                                       allow_pickle=True)["txts"]

                _, x, _ = model.encode(np.repeat(img_emb[np.newaxis, :], batch_size, axis=0),
                                       cond_color, cond_txt, cond_cluster)
                x_txt.extend(model.decode(x, color=cond_color, txt=cond_txt, cluster=cond_cluster))

            if model.cluster_cond_type:
                cond_color = np.repeat(color[np.newaxis, :], 10, axis=0) if model.color_cond_type else None
                cond_txt = np.repeat(txt[np.newaxis, :], 10, axis=0)  if model.txt_cond_type else None
                cond_cluster = []
                for i in range(10):
                    zeros = np.zeros(10)
                    zeros[i]=1
                    cond_cluster.append(zeros)
                _, x, _ = model.encode(np.repeat(img_emb[np.newaxis, :], 10, axis=0),
                                       cond_color, cond_txt, np.array(cond_cluster))
                x_cluster.extend(model.decode(x, color=cond_color, txt=cond_txt, cluster=np.array(cond_cluster)))

            if model.color_cond_type:
                # color exploration
                if model.color_cond_type=="one_hot":
                    cond_color = []
                    #{'green':0,'purple':1,'black':2,'brown':3,'blue':4,'cyan':5,'yellow':6,
                    #'gray':7,'red':8,'pink':9,'orange':10,
                    for i in range(11):
                        zeros = np.zeros(11)
                        zeros[i]=1
                        cond_color.append(zeros)
                    cond_color = np.array(cond_color)
                    color_batchsize = 11
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
                cond_txt = None
                if model.txt_cond_type:
                    cond_txt = np.repeat(txt[np.newaxis, :],  color_batchsize , axis=0)

                    cond_cluster=None
                if model.cluster_cond_type:
                    cond_cluster = np.repeat(cluster[np.newaxis, :],  color_batchsize , axis=0)
                _, x, _ = model.encode(np.repeat(img_emb[np.newaxis, :],  color_batchsize, axis=0),
                                   cond_color, cond_txt, cond_cluster)

                x_color.extend(model.decode(x[:color_batchsize], cond_color, cond_txt, cond_cluster))

        with writer.as_default():
            tf.summary.image( "change_x",
                              spread_image(x_gen,
                                           1*len(colors),
                                           batch_size,img_dim[0],img_dim[1]),
                              step=step)
            tf.summary.image( "latent_walk",
                              spread_image(zs_gen,
                                           1*len(colors),
                                           batch_size,img_dim[0],img_dim[1]),
                              step=step)
            if model.txt_cond_type:
                tf.summary.image( "change_text",
                                  spread_image(x_txt,
                                               1*len(colors),
                                               batch_size,img_dim[0],img_dim[1]),
                                  step=step)
            if model.color_cond_type:
                tf.summary.image( "change_color",
                                  spread_image(x_color,
                                               1*len(colors),
                                               color_batchsize, img_dim[0],img_dim[1]),
                                  step=step)
            if model.cluster_cond_type:
                tf.summary.image( "change_cluster",
                                  spread_image(x_cluster,
                                               1*len(colors),
                                               10, img_dim[0],img_dim[1]),
                                  step=step)
            writer.flush()



def plot_text_condtionals(path, max_len=10, min_len=1):
    txts = np.load(expanduser(path), allow_pickle=True)["txts"]
    txt_len = [len(x) for x in txts if min_len<=len(x)<=max_len ]
    n_bin = max(txt_len)
    print("number of logos: ", len(txt_len))
    _ = plt.hist(txt_len, n_bin)
    plt.ylabel("count")
    plt.xlabel("length")
    plt.show()

path = "./docs/training_plot.csv"
def plot_training_stats(path):
    data =  pd.read_csv(path)
    t=data["Step"]
    with plt.style.context("default"):
        plt.plot(t, data["kl_fake"], label="KLD fake")
        plt.plot(t, data["kl_real"], label="KLD real")
        plt.plot(t, data["kl_rec"], label="KLD rec.")
        plt.plot(t, data["loss_rec"], label="rec. loss")
        plt.plot([0,230000], [110,110], 'b', label="margin")
        plt.ylim(50,500)
        plt.xlim(0,230000)
        plt.ylabel("loss")
        plt.xlabel("training step")
        plt.legend(bbox_to_anchor=(1, 1), loc='bottom left')
        plt.show()


#path = expanduser("./lldboosted_conditionals.npz")
#path_out = expanduser("./plot_cluster_lld.png")
#path_imgs = expanduser("./imgs/")



path = expanduser("~/data/lldboosted_conditionals.npz")
path_out = expanduser("~/data/plot.png")
path_imgs = expanduser("~/data/lld_boosted/")
def plot_cluster(path, path_imgs, path_out):
    x=np.load(path, allow_pickle=True)["res_cluster"]
    #x = [list(a).index(max(a)) for a in np.load(path, allow_pickle=True)["colors_old"]]
    collect = []
    ids = []
    for nr in range(0,10):
        for i,xx in enumerate(x):
            if len(collect)==( nr+1)*10:
                break
            if nr==xx:
                ids.append(i)
                collect.append(io.imread(os.path.join(path_imgs, f"{i}.png"))/255)

    fig, axes= plt.subplots(10, 10)
    for i, a in enumerate(axes):
        for j,b in enumerate(a):
            img = collect[i*10+j]
            b.imshow(img)
            b.axes.xaxis.set_visible(False)
            b.axes.yaxis.set_visible(False)
            b.spines['top'].set_visible(False)
            b.spines['right'].set_visible(False)
            b.spines['bottom'].set_visible(False)
            b.spines['left'].set_visible(False)
    plt.savefig(path_out, quality=100)




def load_model(model_hyperparameter):
    p = params[model_hyperparameter]
    SEED= 27


    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Available gpus: {gpus}")
    if gpus:
        if len(gpus)>=p["gpu"]:
            tf.config.experimental.set_visible_devices(gpus[p["gpu"]], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[p["gpu"]], True)

    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    dataset                  = p["dataset"]
    path_ckpt                = params["dataset"][dataset]['path_ckpt']
    path_cond                = params["dataset"][dataset]['path_cond']
    path_data                = params["dataset"][dataset]['path_data']
    path_log                 = params["dataset"][dataset]['path_log']
    path_spm                 = params["dataset"][dataset]['path_spm']
    path_fid                 = params["dataset"][dataset]["path_fid"]
    path_fid_dataset         = params["dataset"][dataset]["path_fid_dataset"]
    path_inception           = params["dataset"][dataset]["path_inception"]

    restore_model            = p['restore_model']
    img_dim                  = p['img_dim']
    btlnk                    = p['btlnk']
    channels                 = p['channels']
    cond_dim_color           = p['cond_dim_color']
    cond_model               = p['cond_model']
    rnn_dim                  = p['rnn_dim']
    cond_dim_txts            = p['cond_dim_txts']
    cond_dim_clusters        = p['cond_dim_clusters']
    emb_dim                  = p['emb_dim']
    dropout_conditionals     = p['dropout_conditionals']
    dropout_encoder_resblock = p['dropout_encoder_resblock']
    vae_epochs               = p['vae_epochs']
    epochs                   = p['epochs']
    batch_size               = p['batch_size']
    logs_per_epoch           = p['logs_per_epoch']
    weight_rec               = p['weight_rec']
    weight_kl                = p['weight_kl']
    weight_neg               = p['weight_neg']
    weight_aux               = p['weight_aux']
    m_plus                   = p['m_plus']
    lr_enc                   = p['lr_enc']
    lr_dec                   = p['lr_dec']
    beta1                    = p['beta1']
    beta2                    = p['beta2']
    noise_color              = p['noise_color']
    noise_txt                = p['noise_txt']
    noise_img                = p['noise_img']
    txt_len_min              = p["txt_len_min"]
    txt_len_max              = p["txt_len_max"]
    ds_size                  = len([l for l
                                in list(map(len,  np.load(path_cond, allow_pickle=True)["txts"]))
                                if (txt_len_min<=l<=txt_len_max)])
    color_cond_type          = p['color_cond_type']
    cluster_cond_type        = p['cluster_cond_type']
    txt_cond_type            = p['txt_cond_type']
    fid_samples_nr           = p["fid_samples_nr"]
    auxilary                 = p["auxilary"]
    plot_bn                  = p["plot_bn"]
    color_cond_dim           = len(np.load(path_cond, allow_pickle=True)["colors_old" if color_cond_type=="one_hot" else "colors"][1])
    cluster_cond_dim         = 10
    txt_cond_dim             = len(np.load(path_cond, allow_pickle=True)["txts" if txt_cond_type=="rnn" else "txt_embs"][1])
    model_name = p["model_name"]
    if not p["normalizer_enc"]:
        norm = "_NONE"
        normalizer_enc = None
        normalizer_dec = None
    elif p["normalizer_enc"]== "instance":
        norm = "_INST"
        normalizer_enc = tfa.layers.InstanceNormalization
        normalizer_dec = tfa.layers.InstanceNormalization
    elif p["normalizer_enc"]== "group":
        norm = "_GRP"
        normalizer_enc = tfa.layers.GroupNormalization
        normalizer_dec = tfa.layers.GroupNormalization
    elif p["normalizer_enc"]== "batch":
        norm = "_BATCH_"
        normalizer_enc = tf.keras.layers.BatchNormalization
        normalizer_dec = tf.keras.layers.BatchNormalization
    elif p["normalizer_enc"]== "layer":
        norm = "_LAYER"
        normalizer_enc = tf.keras.layers.LayerNormalization
        normalizer_dec = tf.keras.layers.LayerNormalization


    logfrq = ds_size//logs_per_epoch//batch_size
    path_ckpt  = path_ckpt+model_name

    # load sentence piece model
    spm = load_spm(path_spm + ".model")
    spm.SetEncodeExtraOptions("bos:eos") # enable start(=2)/end(=1) symbols
    vocab_dim = spm.vocab_size()

    #pipeline
    bg = batch_cond_spm(path_data, path_cond, spm, batch_size,
                        color_cond_type, txt_cond_type, cluster_cond_type,
                        txt_len_min, txt_len_max)
    data = pipe(lambda: bg, (tf.float32, tf.float32, tf.float32, tf.float32),
                (tf.TensorShape([None, None, None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None])), prefetch=6)
    # model
    model = INTROVAE(img_dim,
                     channels,
                     btlnk,
                     batch_size,
                     cond_dim_color,
                     rnn_dim,
                     cond_dim_txts,
                     cond_dim_clusters,
                     vocab_dim,
                     emb_dim,
                     color_cond_dim,
                     txt_cond_dim,
                     cluster_cond_dim,
                     color_cond_type,
                     txt_cond_type,
                     cluster_cond_type,
                     cond_model,
                     dropout_conditionals=dropout_conditionals,
                     dropout_encoder_resblock=dropout_encoder_resblock,
                     normalizer_enc = normalizer_enc,
                     normalizer_dec = normalizer_dec,
                     weight_rec=weight_rec,
                     weight_kl=weight_kl,
                     weight_neg = weight_neg,
                     weight_aux = weight_aux,
                     m_plus = m_plus,
                     lr_enc= lr_enc,
                     lr_dec= lr_dec,
                     beta1 = beta1,
                     beta2 = beta2,
                     noise_color =noise_color,
                     noise_txt =noise_txt,
                     noise_img =noise_img,
                     auxilary=auxilary)

    # workaround for memoryleak ?
    tf.keras.backend.clear_session()

    #logging
    writer = tf.summary.create_file_writer(pform(path_log, model_name))
    tf.summary.trace_on(graph=True, profiler=True)

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               net=model)


    manager = tf.train.CheckpointManager(ckpt, path_ckpt, checkpoint_name=model_name, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    print(f"RESTORED {model_name}")
    return model, data


def tsne_plot(model_hyperparameter="train_0"):
    model, data= load_model(model_hyperparameter)

    labels, vectors = [], []
    for i in tqdm(range(1000//60)):
        z, mu, lv =model.encode(*next(data))
        labels.extend([0]*60)
        vectors.extend(mu)
    real=len(vectors)
    labels.extend([1]*2000)
    vectors.extend(np.random.normal(size=(
        1000,256)))

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2,  perplexity=50, n_iter=5000)
    tsne_data = tsne.fit_transform(vectors)
    x = [xx[0] for xx in tsne_data[:real]]
    y = [xx[1] for xx in tsne_data[:real]]
    plt.scatter(x,y,s=1, label="img emb")
    x = [xx[0] for xx in tsne_data[real:]]
    y = [xx[1] for xx in tsne_data[real:]]
    plt.scatter(x,y, s=1, label="random")
    plt.legend()
    plt.savefig(expanduser("~/data/tsne_plot.png"), quality=100)
