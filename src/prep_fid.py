import numpy as np
import src.fid as fid
from imageio import imread
import tensorflow.compat.v1 as tf
import os
import glob
from tqdm import tqdm
from os.path import expanduser


def calc_and_save_reference(data_path, output_path, inception_path=None, nr_samples=None):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ########
    # PATHS
    ########
    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    print("check for inception model..", end="\n", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("model ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    image_list = np.array([os.path.join(data_path,fn) for fn in tqdm(os.listdir(data_path))])
    if nr_samples:
        image_list = image_list[:nr_samples]

    print("%d images found and loaded" % len(image_list))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics_from_files(image_list, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")

if __name__=="__main__":
    calc_and_save_reference(data_path=os.path.expanduser("~/data/imgs"),
                            output_path=os.path.expanduser("~/data/fid/mu_var_dataset"),
                            inception_path=os.path.expanduser("~/"))
