import h5py
from collections import Counter
import numpy as np
import concurrent.futures
from tqdm import tqdm
from os.path import expanduser

def get_colors(img, top_n=10):
    #color, count = np.unique(np.around(np.array([colorsys.rgb_to_hsv(*x) for x in np.reshape(np.rollaxis(img,0,3),(160000,3))])/255,2), return_counts=True, axis=0)
    color, count = np.unique(np.around(np.reshape(np.rollaxis(img,0,3),(160000,3))/255,1), return_counts=True, axis=0)
    order = count.argsort()[-top_n:]
    return color[order].flatten()


path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
imgs = h5py.File(path_data, 'r')['data'][:100]

### multicore if imgs can be stored in memory
with concurrent.futures.ProcessPoolExecutor() as executor:
    color_conditional = list(tqdm(executor.map(get_colors, imgs), total=len(imgs)))
np.savez_compressed(expanduser("~/data/color_conditional.npz"), {"colors": color_conditional})

### non multicore
#color_conditional = []
#for img in tqdm(imgs):
#    color_conditional.append(get_colors(img))
