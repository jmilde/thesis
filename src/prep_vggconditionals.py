import tensorflow as tf
import os
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from src.hyperparameter import params
from src.util_sp import load_spm
from src.models.introvae import INTROVAE
from src.util_io import pform
from collections import Counter

def add_resnet_condtionals(path_imgs, path_conditionals):
    with tf.device('/CPU:0'):
        model = tf.keras.applications.VGG16(input_shape=(128,128,3),
                                            include_top=False,
                                            weights="imagenet")

        embeddings = []
        for img in tqdm(range(len(os.listdir(path_imgs)))):
            x = io.imread(os.path.join(path_imgs, str(img)+".png"))/255
            embeddings.append(model(np.array([x])).numpy().flatten())
    print("start clustering")
    kmeans = KMeans(n_clusters=10).fit(embeddings)
    print("load existing conditionals")
    print(Counter(kmeans.labels_).most_common())

    x = np.load(path_conditionals, allow_pickle=True)

    print("save conditionals")
    np.savez_compressed(path_conditionals,
                        colors=x["colors"],
                        txts=x["txts"],
                        txt_embs = x["txt_embs"],
                        colors_old = x["colors_old"],
                        res_cluster= kmeans.labels_,
                        vae_cluster=x["vae_cluster"])
    print("done")

def vae_embeddings(path_imgs, path_conditionals, model_name, hyperparameter):
    p = params[hyperparameter]
    SEED= 27

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

    normalizer_enc = tf.keras.layers.BatchNormalization
    normalizer_dec = tf.keras.layers.BatchNormalization




    path_ckpt  = path_ckpt+model_name

    # load sentence piece model
    spm = load_spm(path_spm + ".model")
    spm.SetEncodeExtraOptions("bos:eos") # enable start(=2)/end(=1) symbols
    vocab_dim = spm.vocab_size()

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
    print(f"\nmodel: {model_name} restored\n")

    conds = np.load(os.path.expanduser("~/data/lldboosted_conditionals.npz"), allow_pickle=True)
    with tf.device('/CPU:0'):

        embeddings = []
        for img in tqdm(range(len(os.listdir(path_imgs)))):
            x = io.imread(os.path.join(path_imgs, str(img)+".png"))/255
            _, x, _  = model.encode(np.array([x]))
            embeddings.append(x.numpy().flatten())

    print("start clustering")
    kmeans = KMeans(n_clusters=10).fit(embeddings)
    print("load existing conditionals")
    x = np.load(path_conditionals, allow_pickle=True)

    print("save conditionals")
    np.savez_compressed(expanduser("~/data/eudata_conditionals.npz"),
                        colors=x["colors"],
                        txts=x["txts"],
                        txt_embs = x["txt_embs"],
                        colors_old = x["colors_old"],
                        vae_cluster = [0],
                        res_cluster=  x["res_cluster"])




def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Available gpus: {gpus}")
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
    print("lld dataset")
    path_imgs = os.path.expanduser("~/data/lld_boosted/")
    path_conditionals = os.path.expanduser("~/data/lldboosted_conditionals.npz")
    add_resnet_condtionals(path_imgs, path_conditionals)


    path_imgs = os.path.expanduser("~/data/imgs/")
    path_conditionals = os.path.expanduser("~/data/eudata_conditionals.npz")
    add_resnet_condtionals(path_imgs, path_conditionals)


if __name__=="__main__":
    main()
