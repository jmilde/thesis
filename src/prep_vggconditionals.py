import tensorflow as tf
import os
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def add_resnet_condtionals(path_imgs, path_conditionals):
    with tf.device('/CPU:0'):
        model = tf.keras.applications.VGG16(input_shape=(128,128,3),
                                            include_top=False,
                                            weights="imagenet")

        embeddings = []
        for img in tqdm(os.listdir(path_imgs)):
            x = io.imread(os.path.join(path_imgs, img))/255
            embeddings.append(model(np.array([x])).numpy().flatten())

    print("start clustering")
    kmeans = KMeans(n_clusters=10).fit(embeddings)
    print("load existing conditionals")
    x = np.load(path_conditionals, allow_pickle=True)

    print("save conditionals")
    np.savez_compressed(path_conditionals,
                        colors=x["colors"],
                        txts=x["txts"],
                        txt_embs = x["txt_embs"],
                        colors_old = x["colors_old"],
                        res_cluster= kmeans.labels_)
    print("done")

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)

    path_imgs = os.path.expanduser("~/data/imgs/")
    path_conditionals = os.path.expanduser("~/data/eudata_conditionals.npz")
    add_resnet_condtionals(path_imgs, path_conditionals)

    print("lld dataset")
    path_imgs = os.path.expanduser("~/data/lld_boosted/")
    path_conditionals = os.path.expanduser("~/data/lldboosted_conditionals.npz")
    add_resnet_condtionals(path_imgs, path_conditionals)


if __name__=="__main__":
    main()
