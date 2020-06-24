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


def main(path_data, resize_size, path_out, batch_size):
    """
    preprocesses the data and saves it as a .npz file with the keys: imgs, colors, txts
    """
    print("collecting all paths...")
    paths = np.array(collect_paths(path_data))
    resize_size = np.repeat([resize_size], len(paths), axis=0)

    print("processing the data")
    #data=[prep(a,b) for a,b in tqdm(zip(paths, resize_size), total=len(paths))] # for debugging

    for i, (paths_, resize_size_) in enumerate(zip(np.split(paths, batch_size), np.split(resize_size, batch_size))):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            data = list(tqdm(executor.map(prep, paths_, resize_size_), total=len(paths_)))


        data = [d for d in data if d]
        print("getting images")
        imgs = np.array([d[0] for d in data], dtype="uint8")
        print("getting texts")
        txts = [d[1] for d in data]
        print("getting colors")
        colors = np.array([d[2] for d in data], dtype="float32")
        print(f"saving to part to {path_out}eudata_prep{i}.npz")
        np.savez_compressed(os.path.join(path_out, f"eudata_prep_pt{i}.npz"), imgs=imgs, colors=colors, txts=txts )


    print("final step: saving images")
    id_nr = 0
    colors, txts = [], []
    for i in range(batch_size):
        data = np.load(os.path.join(path_out, f"eudata_prep_pt1.npz"), allow_pickle=True)
        for i in tqdm(data["imgs"]):
            imsave(os.path.join(path_out, f"imgs/{id_nr}.png"), i)
            id_nr+=1
        colors.extend(data["colors"])
        txts.extend(data["txts"])
    np.savez_compressed(os.path.join(path_out, f"eudata_conditionals.npz"), colors=colors, txts=txts)
    print(f"Done: saved all images in {path_out}imgs/ and the conditionals as eudata_conditionals.npz")


def prep(path, resize_size):
    with open(path) as fd:
        try:
            doc = xmltodict.parse(fd.read())
            infos = doc["Transaction"]["TradeMarkTransactionBody"]["TransactionContentDetails"]["TransactionData"]["TradeMarkDetails"]["TradeMark"]
            if infos.get("MarkFeature")=="Figurative":
                #nr = infos.get("ApplicationNumber") # application number
                #if nr in labels: print("WARNING: DUPLICATES")

                #application_date = infos.get("ApplicationDate")

                #mark_description = infos.get("MarkDescriptionDetails", {}).get("MarkDescription")
                #mark_description_en = get_description([mark_description]) if mark_description and type(mark_description)!=list else get_description(mark_description)

                #displayed_text = infos.get("WordMarkSpecification", {}).get("MarkVerbalElementText")

                ### option to use/acess vienna and nice classification info
                #vienna_classification = infos.get("MarkImageDetails", {}).get("MarkImage", {}).get("MarkImageCategory", {}).get("CategoryCodeDetails", {}).get("CategoryCode")
                #nice_version = infos.get("GoodsServicesDetails", {}).get("GoodsServices", {}).get("ClassificationVersion")
                #nice_infos = infos.get("GoodsServicesDetails", {}).get("GoodsServices", {}).get("ClassDescriptionDetails", {}).get("ClassDescription", {})
                #if nice_infos and type(nice_infos)!=list: nice_infos = [nice_infos]
                #nice_classification_en = {x.get("ClassNumber"): (get_description([x.get("GoodsServicesDescription")])
                #                                                           if type(x.get("GoodsServicesDescription"))!=list and x.get("GoodsServicesDescription")
                #                                                            else get_description(x.get("GoodsServicesDescription")) )
                #                                     for x in nice_infos}


                img_path = infos.get("MarkImageDetails", {}).get("MarkImage", {}).get("MarkImageURI")

                if img_path:
                    img = io.imread(os.path.join(path_data, img_path.split("//")[1]))
                    shape = img.shape

                    if len(shape)==3 and shape[-1]>3: # skip over faulty images
                        return None

                    # resize image while keeping aspect ratio
                    ratio = min(resize_size[0]/shape[0], resize_size[1]/shape[1])
                    img_resized = resize(img, (int(shape[0]*ratio), int(shape[1]*ratio)))

                    # transform grayscale images to rgb
                    if len(img_resized.shape)==2:
                        img_resized = gray2rgb(img_resized)
                    #print(os.path.join(path_data, img_path.split("//")[1]), img_resized.shape)

                    # pad with white pixels so that the image is centered
                    pad_x = (resize_size[0]-img_resized.shape[0])
                    pad_x = (math.ceil(pad_x/2), math.floor(pad_x/2))
                    pad_y = (resize_size[1]-img_resized.shape[1])
                    pad_y = (math.ceil(pad_y/2), math.floor(pad_y/2))
                    paddings = (pad_x,pad_y,(0,0))
                    img_resized = np.pad(img_resized, paddings, constant_values=1)*255

                    displayed_text = infos.get("WordMarkSpecification", {}).get("MarkVerbalElementText")
                    colors = get_colors(img_resized)
                    return img_resized.astype("uint8"), displayed_text if displayed_text else "", colors.astype("float32")
        except Exception as e:
            print(f"{e}: {path}, {img_path}")
        return None

if __name__=="__main__":
    path_data   = "../eudata_unpacked/"
    path_out    = "../"
    batch_size=3 # batch size 2 needs ~ 135gb RAM
    resize_size = (256,256,3)


    if not os.path.isdir(os.path.join(path_out, "imgs")):
        os.mkdir(os.path.join(path_out, "imgs"))
    main(path_data, resize_size, path_out, batch_size)
