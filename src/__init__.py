from skimage.io import imsave
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from os.path import expanduser
from src.analyze_introvae import load_model
from src.util_sp import load_spm
import ast
import numpy as np
import tensorflow as tf
import datetime
from src.hyperparameter import params

app = Flask(__name__)

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
    app.spm = load_spm(expanduser("~/data/logo_vocab") + ".model")
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

    imgs = app.generator.decode(img_embs)#, color_cond, txt_cond)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    for i, (emb, img) in enumerate(zip(img_embs, imgs),1):
        print(img)
        imsave(f"./src/static/imgtmp{i}.jpg", np.clip(img*255, a_max=255, a_min=0).astype("uint8"))
        np.save(f"./src/static/embtmp{i}.npy", emb)

    return jsonify({
        "img_path1":f"../static/imgtmp1.jpg?{time}",
        "img_path2":f"../static/imgtmp2.jpg?{time}",
        "img_path3":f"../static/imgtmp3.jpg?{time}",
        "img_path4":f"../static/imgtmp4.jpg?{time}",
        "img_path5":f"../static/imgtmp5.jpg?{time}",
        "img_path6":f"../static/imgtmp6.jpg?{time}",
        "img_path7":f"../static/imgtmp7.jpg?{time}",
        "img_path8":f"../static/imgtmp8.jpg?{time}",
        "img_path9":f"../static/imgtmp9.jpg?{time}"})

@app.route('/generate_similar', methods=['POST'])
def generate_similar():
    data = request.json

    ### get embeddings
    img_nr = data['img']
    og_img = np.load(f"./src/static/embtmp{img_nr}.npy")
    ### get text
    text = data.get("text")
    txt_cond = np.array(app.spm.encode(text))
    txt_cond = np.repeat(txt_cond[np.newaxis, :], 9, axis=0)
    ### get colors:
    color_cond = np.array(prep_colors(data))
    color_cond = np.repeat(color_cond[np.newaxis, :], 9, axis=0)

    img_embs= np.repeat(og_img[np.newaxis, :], 9, axis=0) +np.random.normal(0, scale=0.3, size=(9,params["for_flask"]["btlnk"]))
    #img_embs[img_embs<0]=0
    #img_embs[img_embs>1]=1
    #img_embs[img_nr]=og_img # keep the image we clicked on the same


    imgs = app.generator.decode(img_embs)#, color_cond, txt_cond)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    for i, (emb, img) in enumerate(zip(img_embs, imgs),1):
        if i != img_nr:
            imsave(f"./src/static/imgtmp{i}.jpg", np.clip(img*255, a_max=255, a_min=0).astype("uint8"))
            np.save(f"./src/static/embtmp{i}.npy", emb)

    return jsonify({
        "img_path1":f"../static/imgtmp1.jpg?{time}",
        "img_path2":f"../static/imgtmp2.jpg?{time}",
        "img_path3":f"../static/imgtmp3.jpg?{time}",
        "img_path4":f"../static/imgtmp4.jpg?{time}",
        "img_path5":f"../static/imgtmp5.jpg?{time}",
        "img_path6":f"../static/imgtmp6.jpg?{time}",
        "img_path7":f"../static/imgtmp7.jpg?{time}",
        "img_path8":f"../static/imgtmp8.jpg?{time}",
        "img_path9":f"../static/imgtmp9.jpg?{time}"})
    # either spread image or do sth else
