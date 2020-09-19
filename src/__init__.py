from skimage.io import imsave
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from os.path import expanduser
from src.util_sp import load_spm
import ast
import numpy as np
import tensorflow as tf
import datetime
from src.hyperparameter import params
from os.path import expanduser
from src.util_tf import batch_cond_spm, pipe, spread_image
from src.util_io import pform
from src.models.introvae import INTROVAE
from src.analyze_introvae import run_tests, calculate_scores
from src.util_sp import load_spm
import numpy as np
import h5py
from tqdm import trange,tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import os
from os.path import expanduser
from src.hyperparameter import params


app = Flask(__name__)

def load_model():
    p = params["for_flask"]
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
    start_step = p["start_step"]
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
    return model


def prep_colors(data):
    value_sum = 0
    collector = defaultdict(lambda: defaultdict())
    for key in data:
        try:
            typ,nr= key.split("_")
            if typ=="value":
                if data[key]:
                    value_sum += int(data[key])
                    collector[nr][typ]=int(data[key])
                else:
                    del collector[nr] # only if both color and value are filled
            if typ=="color":
                color= data[key]
                if color:
                    collector[nr][typ]= ast.literal_eval(color.replace("rgb", ""))
                else:
                    del collector[nr] # only if both color and value are filled
        except: continue

    # process the color conditionals
    normalize = 1/100
    color_cond = np.full([4,4,4],0, dtype=np.float32) # each datapoint represents a color
    if value_sum>100: normalize = (100/value_sum)/100 # normalize from 0-100 to 0-1
    elif value_sum<100: color_cond[3,3,3]= (100-value_sum)/100

    for k,v in collector.items():
        color = np.around(np.array(v["color"])/85.33333333333333,0).astype("int")
        color_cond[color[0], color[1], color[2]]=v["value"]*normalize
    return color_cond.flatten()

@app.before_first_request
def load_models():
    app.spm = load_spm("/home/jan/data/logo_vocab" + ".model")
    app.spm.SetEncodeExtraOptions("bos:eos")
    #app.predictor = tf.keras.models.load_model('~/models/testmodel_save')
    app.generator= load_model()


@app.route("/")
def index():
    return render_template('index.html',
                           text   ="Beispieltext",
                           color_1="rgb(0,0,1)",
                           value_1="40")


@app.route('/generate_random', methods=['POST'])
def generate_random():
    data = request.json

    ### get text
    text = data.get("text")
    txt_cond = np.array(app.spm.encode(text))
    txt_cond = np.repeat(txt_cond[np.newaxis, :], 9, axis=0)

    ### get colors:
    color_cond = np.array(prep_colors(data))
    color_cond = np.repeat(color_cond[np.newaxis, :], 9, axis=0)
    ### random image embedding
    img_embs = np.random.normal(0,1,(9,params["for_flask"]["btlnk"]))
    imgs = app.generator.decode(img_embs, color_cond, txt_cond)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    for i, (emb, img) in enumerate(zip(img_embs, imgs),1):
        imsave(expanduser(f"./src/static/imgtmp{i}.jpg"), np.clip(img*255, a_max=255, a_min=0).astype("uint8"))
        np.save(expanduser(f"./src/static/embtmp{i}.npy"), emb)

    return jsonify({
        "img_path1":expanduser(f"../static/imgtmp1.jpg?{time}"),
        "img_path2":expanduser(f"../static/imgtmp2.jpg?{time}"),
        "img_path3":expanduser(f"../static/imgtmp3.jpg?{time}"),
        "img_path4":expanduser(f"../static/imgtmp4.jpg?{time}"),
        "img_path5":expanduser(f"../static/imgtmp5.jpg?{time}"),
        "img_path6":expanduser(f"../static/imgtmp6.jpg?{time}"),
        "img_path7":expanduser(f"../static/imgtmp7.jpg?{time}"),
        "img_path8":expanduser(f"../static/imgtmp8.jpg?{time}"),
        "img_path9":expanduser(f"../static/imgtmp9.jpg?{time}")})



@app.route('/generate_similar', methods=['POST'])
def generate_similar():
    data = request.json

    ### get embeddings
    img_nr = data['img']
    og_img = np.load(expanduser(f"./src/static/embtmp{img_nr}.npy"))
    ### get text
    text = data.get("text")
    txt_cond = np.array(app.spm.encode(text))
    txt_cond = np.repeat(txt_cond[np.newaxis, :], 9, axis=0)
    ### get colors:
    color_cond = np.array(prep_colors(data))
    print(color_cond)
    color_cond = np.repeat(color_cond[np.newaxis, :], 9, axis=0)

    img_embs= np.repeat(og_img[np.newaxis, :], 9, axis=0) +np.random.normal(0, scale=0.3, size=(9,params["for_flask"]["btlnk"]))
    #img_embs[img_embs<0]=0
    #img_embs[img_embs>1]=1
    #img_embs[img_nr]=og_img # keep the image we clicked on the same


    imgs = app.generator.decode(img_embs, color_cond, txt_cond)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    for i, (emb, img) in enumerate(zip(img_embs, imgs),1):
        if i != img_nr:
            imsave(expanduser(f"./src/static/imgtmp{i}.jpg"), np.clip(img*255, a_max=255, a_min=0).astype("uint8"))
            np.save(expanduser(f"./src/static/embtmp{i}.npy"), emb)

    return jsonify({
        "img_path1":expanduser(f"../static/imgtmp1.jpg?{time}"),
        "img_path2":expanduser(f"../static/imgtmp2.jpg?{time}"),
        "img_path3":expanduser(f"../static/imgtmp3.jpg?{time}"),
        "img_path4":expanduser(f"../static/imgtmp4.jpg?{time}"),
        "img_path5":expanduser(f"../static/imgtmp5.jpg?{time}"),
        "img_path6":expanduser(f"../static/imgtmp6.jpg?{time}"),
        "img_path7":expanduser(f"../static/imgtmp7.jpg?{time}"),
        "img_path8":expanduser(f"../static/imgtmp8.jpg?{time}"),
        "img_path9":expanduser(f"../static/imgtmp9.jpg?{time}")})
    # either spread image or do sth else
