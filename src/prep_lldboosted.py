from collections import defaultdict
from skimage import io
from skimage.color import gray2rgb
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
import concurrent.futures
import math
import numpy as np
import os
import xmltodict
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import webcolors
from os.path import expanduser
from sentence_transformers import SentenceTransformer

### value to position
#x=[[a,b,c] for a in range(0,256,85) for b in range(0,256,85) for c in range(0,256,85)]
#[f"rgb({b},{a},{c})" for a,b,c  in x]
color2nr = {'green':0,
            'purple':1,
            'black':2,
            'brown':3,
            'blue':4,
            'cyan':5,
            'yellow':6,
            'gray':7,
            'red':8,
            'pink':9,
            'orange':10,
            }

def get_shade(color_name):
    if color_name == 'darkolivegreen' or color_name == 'olive' or color_name == 'olivedrab' or color_name == 'yellowgreen' or color_name == 'limegreen' or color_name == 'lime' or color_name == 'lawngreen' or color_name == 'chartreuse' or color_name == 'greenyellow' or color_name == 'springgreen' or  color_name == 'mediumspringgreen' or color_name == 'lightgreen' or color_name == 'palegreen' or color_name == 'darkseagreen' or color_name == 'mediumaquamarine' or  color_name == 'mediumseagreen' or color_name == 'seagreen' or color_name == 'forestgreen' or color_name == 'green' or color_name == 'darkgreen':
        shade =  'green'
    elif color_name == 'lavender' or color_name == 'thistle' or color_name == 'plum' or color_name == 'violet' or color_name == 'orchid' or color_name == 'fuchsia' or color_name == 'magenta' or color_name == 'mediumorchid' or color_name == 'mediumpurple' or color_name == 'blueviolet' or  color_name == 'darkviolet' or color_name == 'darkorchid' or color_name == 'darkmagenta' or color_name == 'purple' or color_name == 'indigo' or  color_name == 'darkslateblue' or color_name == 'slateblue' or color_name == 'mediumslateblue':
        shade =  'purple'
    elif color_name == 'white' or color_name == 'snow' or color_name == 'honeydew' or color_name == 'mintcream' or color_name == 'azure' or color_name == 'aliceblue' or color_name == 'ghostwhite' or color_name == 'whitesmoke' or color_name == 'seashell' or color_name == 'beige' or  color_name == 'oldlace' or color_name == 'floralwhite' or color_name == 'ivory' or color_name == 'aquawhite' or color_name == 'linen' or  color_name == 'lavenderblush' or color_name == 'mistyrose' or color_name == 'antiquewhite':
        shade =  'white'
    elif color_name == 'cornsilk' or color_name == 'blanchedalmond' or color_name == 'bisque' or color_name == 'navajowhite' or color_name == 'wheat' or color_name == 'burlywood' or color_name == 'tan' or color_name == 'rosybrown' or color_name == 'sandybrown' or color_name == 'goldenrod' or  color_name == 'darkgoldenrod' or color_name == 'peru' or color_name == 'chocolate' or color_name == 'saddlebrown' or color_name == 'sienna' or  color_name == 'brown' or color_name == 'maroon':
        shade =  'brown'
    elif color_name == 'lightsteelblue' or color_name == 'powderblue' or color_name == 'lightblue' or color_name == 'skyblue' or color_name == 'lightskyblue' or color_name == 'deepskyblue' or color_name == 'dodgerblue' or color_name == 'cornflowerblue' or color_name == 'steelblue' or color_name == 'royalblue' or  color_name == 'blue' or color_name == 'mediumblue' or color_name == 'darkblue' or color_name == 'navy' or color_name == 'midnightblue':
        shade =  'blue'
    elif color_name == 'aqua' or color_name == 'cyan' or color_name == 'lightcyan' or color_name == 'paleturquoise' or color_name == 'aquamarine' or color_name == 'turquoise' or color_name == 'mediumturquoise' or color_name == 'darkturquoise' or color_name == 'lightseagreen' or color_name == 'cadetblue' or  color_name == 'darkcyan' or color_name == 'teal':
        shade =  'cyan'
    elif color_name == 'yellow' or color_name == 'lightyellow' or color_name == 'lemonchiffon' or color_name == 'lightgoldenrodyellow' or color_name == 'papayawhip' or color_name == 'moccasin' or color_name == 'peachpuff' or color_name == 'palegoldenrod' or color_name == 'khaki' or color_name == 'darkkhaki' or  color_name == 'gold':
        shade =  'yellow'
    elif color_name == 'gainsboro' or color_name == 'lightgrey' or color_name == 'silver' or color_name == 'darkgrey' or color_name == 'grey' or color_name == 'dimgrey' or color_name == 'lightslategrey' or color_name == 'darkslategrey' or color_name == 'slategrey':
        shade =  'gray'
    elif color_name == 'lightgray' or color_name == 'darkgray' or color_name == 'gray' or color_name == 'dimgray' or color_name == 'lightslategray' or color_name == 'darkslategray' or color_name == 'slategray':
        shade =  'gray'
    elif color_name == 'lightsalmon' or color_name == 'salmon' or color_name == 'darksalmon' or color_name == 'lightcoral' or color_name == 'indianred' or color_name == 'crimson' or color_name == 'firebrick' or color_name == 'darkred' or color_name == 'red' :
        shade =  'red'
    elif color_name == 'pink' or color_name == 'lightpink' or color_name == 'hotpink' or color_name == 'deeppink' or color_name == 'palevioletred' or color_name == 'mediumvioletred':
        shade = 'pink'
    elif color_name == 'orangered' or color_name == 'tomato' or color_name == 'coral' or color_name == 'darkorange' or color_name == 'orange':
        shade = 'orange'
    elif color_name == 'black':
        shade = 'black'
    else:
        shade = 'unknown'
    return shade


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def unison_shfl(l1, l2, SEED=25):
    l1_shfl, l2_shfl = [], []
    shfl_idx = np.arange(len(l1))
    np.random.seed(SEED)
    np.random.shuffle(shfl_idx)
    for i in shfl_idx:
        l1_shfl.append(l1[i])
        l2_shfl.append(l2[i])
    return l1_shfl, l2_shfl


def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def get_colors_old(img):
    """output example: 3 colors(centroids) and percentage
    (array([
     [ 11.68018018,  14.74774775,  16.40315315],
     [244.57594937, 245.53375527, 245.44092827],
     [119.30412371, 124.46563574, 124.33161512]]),
    array([0.31567383, 0.29217529, 0.39215088]))
"""
    try:
        img = img.reshape((img.shape[0] * img.shape[1],3))
        clt = MiniBatchKMeans(n_clusters=3)
        clt.fit(img)
        histogram = find_histogram(clt)
        order = sorted(range(len(histogram)), key=histogram.__getitem__)
        hist, cluster_centers = [],[]
        for i in order:
            hist.append(histogram[i])
            cluster_centers.append(clt.cluster_centers_[i])

        color = get_shade(get_colour_name(cluster_centers[-1]))
        if (color =="white" or color=="unknown") and len(order)>1:
            color = get_shade(get_colour_name(cluster_centers[-2]))
        if (color =="white" or color=="unknown") and len(order)>2:
            color = get_shade(get_colour_name(cluster_centers[-3]))
        if color !="white" and color!="unknown":
            output = np.zeros(11)
            output[color2nr[color]]=1
            return output
        return ""

    except Exception as e:
        print(e)
        print(img.shape)
        print(histogram)
        print(clt.cluster_centers_)
        print(color)
        return ""


def get_description(description, language_code="en"):
    if description:
        if type(description)==list:
            for x in description:
                if x.get("@languageCode")==language_code:
                    return x.get("#text")
        else:
            if description.get("@languageCode")==language_code:
                return description.get("#text")

    return None

def get_colors(img):
    """
    /!\ images need to be in rgb 255 format

    idea:
    6-bit RGB palette use 2 bits for each of the red, green, and blue color components.
    (see wikipedia for image example)
    This results in a (2²)³ = 4³ = 64-color palette"""
    out = np.full([4,4,4],0, dtype=np.float32) # each datapoint represents a color
    img_dim = img.shape[0]*img.shape[1]
    colors, counts = np.unique(np.around((np.reshape(img,(img_dim,3))/85.33333333333333),0).astype("uint8"),return_counts=True, axis=0) # 2² per color channel
    for color, count in zip(colors, counts):
        out[color[0],color[1],color[2]]= round(count/img_dim,2)
    return out.flatten()


def collect_paths(path_data):
    return [os.path.join(path, f) for path, dirs, files in tqdm(os.walk(path_data), total= 699399) for f in files if os.path.splitext(f)[1]==".xml"]


def main(path_data, path_data_lld, path_data_metu, path_lbls_metu, resize_size, path_out, batch_size):
    """
    preprocesses the data and saves it as a .npz file with the keys: imgs, colors, txts
    """


    batch_nr = 0

    ##############################
    # LARGE LOGO DATASET LOGANv2 #
    ##############################
    print("PROCESSING LLD")
    paths = [os.path.join(path_data_lld, img) for img in os.listdir(path_data_lld)]
    resize_size_ = np.repeat([resize_size], len(paths), axis=0)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = list(tqdm(executor.map(prep_2, paths, resize_size_), total=len(paths)))
    data = [d for d in data if d]
    print("getting images of LLD")
    imgs = np.array([d[0] for i,d in enumerate(data)], dtype="uint8")
    print("getting texts of LLD")
    txts = [d[1] for i,d in enumerate(data)]
    print("getting colors of LLD")
    colors = np.array([d[2] for i,d in enumerate(data)], dtype="float32")
    print("getting OH-colors of LLDBOOSTED")
    colors_old = np.array([d[3] for i,d in enumerate(data)], dtype="float32")
    print(f"saving to part to {path_out}lldboosted_prep_pt{batch_nr}.npz")
    np.savez_compressed(os.path.join(path_out, f"lldboosted_prep.npz"),
                        imgs=imgs, colors=colors, txts=txts, colors_old=colors_old)

    print("final step: saving images")
    id_nr = 0
    colors, colors_old, txts = [], [], []
    data = np.load(os.path.join(path_out, f"lldboosted_prep.npz"), allow_pickle=True)
    for j in tqdm(data["imgs"]):
        imsave(os.path.join(path_out, f"lld_boosted/{id_nr}.png"), j)
        id_nr+=1
    colors.extend(data["colors"])
    colors_old.extend(data["colors_old"])
    txts.extend(data["txts"])

    # calculate bert sentence embeddings
    model = SentenceTransformer("distiluse-base-multilingual-cased")
    txt_embs = model.encode(np.array(txts))



    np.savez_compressed(os.path.join(path_out, f"lldboosted_conditionals.npz"),
                        colors=colors,
                        txts=txts,
                        txt_embs = txt_embs,
                        colors_old=np.array(colors_old))
    print(f"Done: saved all images in {path_out}imgs/ and the conditionals as lldboosted_conditionals.npz")


def prep_2(path, resize_size):
    try:
        img = io.imread(path)
    except:
        return None

    shape = img.shape

    if ((len(shape)==3) and (shape[-1]>3))or((shape[0]<resize_size[0]) and (shape[1]<resize_size[0])): # skip over faulty images
        return None

    if len(img.shape)<3:
        img = gray2rgb(img)
    ratio = min(resize_size[0]/shape[0], resize_size[1]/shape[1])
    img_resized = resize(img, (int(shape[0]*ratio), int(shape[1]*ratio)))*255
    colors = get_colors(img_resized)

    pad_x = (resize_size[0]-img_resized.shape[0])
    pad_x = (math.ceil(pad_x/2), math.floor(pad_x/2))
    pad_y = (resize_size[1]-img_resized.shape[1])
    pad_y = (math.ceil(pad_y/2), math.floor(pad_y/2))
    paddings = (pad_x,pad_y,(0,0))
    img_resized = np.pad(img_resized, paddings, constant_values=1)
    colors_oh = get_colors_old(img_resized)
    if len(colors_oh)==11:
        return img_resized.astype("uint8"), "", colors.astype("float32"), colors_oh
    return None

if __name__=="__main__":
    path_data      = expanduser("../data/eudata_unpacked/")
    path_data_lld  = expanduser("../data/LOGOS_REFORMAT/")
    path_data_metu = expanduser("../data/930k_logo_v3")
    path_lbls_metu = expanduser("../data/METU_logo_type_info.csv")
    path_out       = expanduser("../data/")
    batch_size     = 3 # batch size 2 needs ~ 135gb RAM
    resize_size    = (128,128,3)

    if not os.path.isdir(os.path.join(path_out, "lld_boosted")):
        os.mkdir(os.path.join(path_out, "lld_boosted"))
    main(path_data, path_data_lld, path_data_metu, path_lbls_metu, resize_size, path_out, batch_size)
