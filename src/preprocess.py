import h5py
from collections import Counter
import numpy as np
import concurrent.futures
from tqdm import tqdm
from os.path import expanduser
from skimage.transform import resize


#################################
# MOST FREQUENT COLORS IN IMAGE #
#################################
# idea:
# 6-bit RGB palette use 2 bits for each of the red, green, and blue color components.
# This results in a (2²)³ = 4³ = 64-color palette
# problems: maybe a lot of the colors will be hues of white, what to do if someone selects only 1 color in inference
#           -> maybe only add colors when the number is above a certain threshold, if only 3 numbers are high enoguh, then the rest will be filled with "white"" (1,1,1)
#
# options: use hsv instead of rgb

def get_colors(img, shape):
    out = np.full([4,4,4],0, dtype=np.float32)
    img = img[:,:shape[1], :shape[2]]
    img = np.rollaxis(img,0,3)
    img_dim = img.shape[0]*img.shape[1]
    colors, counts = np.unique(np.around((np.reshape(img,(img_dim,3))/85.33333333333333),0).astype("uint8"),return_counts=True, axis=0) # 2² per color channel
    for color, count in zip(colors, counts):
        out[color[0],color[1],color[2]]= round(count/img_dim,2)
    return out.flatten()

def prep_color():
    path_data = expanduser('~/data/LLD-logo.hdf5')
    path_out  = expanduser("~/data/color_conditional.npz")

    #path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
    len_imgs = len(h5py.File(path_data, 'r')['shapes'])
    imgs = iter(h5py.File(path_data, 'r')['data'])
    shapes = iter(h5py.File(path_data, 'r')['shapes'])
    #ds = iter([[img, shape] for img, shape in zip(imgs, shapes)])

    ### multicore if imgs can be stored in memory
    with concurrent.futures.ProcessPoolExecutor() as executor:
        color_conditional = list(tqdm(executor.map(get_colors, imgs, shapes), total=122920))
    np.savez_compressed(expanduser(path_out), colors= color_conditional)
    print("saved preprocessed colors")

    ### non multicore
        #color_conditional = []
        #for img in tqdm(imgs):
        #    color_conditional.append(get_colors(img))

def resize_img(img, shape):
    if shape[1]==400:
        return resize(np.rollaxis(img,0,3), (384,384)).astype("uint8")
    else:
        return resize(np.rollaxis(img[:,:shape[1], :shape[2]],0,3), (384,384)).astype("uint8")

def prep_resize():
    path_data = expanduser('~/data/LLD-logo.hdf5')
    path_out  = expanduser("~/data/imgs_resized.npz")

    imgs = h5py.File(path_data, 'r')['data']
    shapes = h5py.File(path_data, 'r')['shapes']
    with concurrent.futures.ProcessPoolExecutor() as executor:
        imgs_resized = list(tqdm(executor.map(resize_img, imgs, shapes), total=len(imgs)))
    np.savez_compressed(expanduser(path_out), imgs= imgs_resized)


if __name__=="__main__":
    #prep_data()
    prep_color()
