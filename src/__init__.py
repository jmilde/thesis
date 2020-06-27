from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from collections import defaultdict
import ast
from src.util_sp import load_spm
from os.path import expanduser


app = Flask(__name__)


@app.before_first_request
def load_models():
    app.spm = load_spm(expanduser("~/data/logo_vocab") + ".model")
    app.spm.SetEncodeExtraOptions("bos:eos")
    app.predictor = tf.keras.models.load_model('./data/testmodel_save')


@app.route("/")
def index():
    return render_template('index.html',
                           text   ="Beispieltext",
                           color_1="rgb(0,0,1)",
                           value_1="40")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    ### get text
    text = data.get("text")
    txt_cond = app.spm.encode(text)
    ### get colors:
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


    print(txt_cond)
    #predictions = app.predictor.generate(img_emb, color_cond, txt_cond)

    return render_template('index.html',
                           text   =data.get("text"),
                           color_1=data.get("color_1"),
                           color_2=data.get("color_2"),
                           color_3=data.get("color_3"),
                           color_4=data.get("color_4"),
                           color_5=data.get("color_5"),
                           value_1=data.get("value_1"),
                           value_2=data.get("value_2"),
                           value_3=data.get("value_3"),
                           value_4=data.get("value_4"),
                           value_5=data.get("value_5"))
