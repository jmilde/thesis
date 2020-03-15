import h5py
from collections import Counter
import numpy as np
import concurrent.futures
from tqdm import tqdm
from os.path import expanduser


#################################
# MOST FREQUENT COLORS IN IMAGE #
#################################
# idea: norm the rgb values between 0-1 and round them to get a rough distribution
#       then report the n_top most frequent ones
#       later on in the interface people could select up to 10 colors in a table like here https://www.rapidtables.com/web/color/RGB_Color.html
#
# problems: maybe a lot of the colors will be hues of white, what to do if someone selects only 1 color in inference
#           -> maybe only add colors when the number is above a certain threshold, if only 3 numbers are high enoguh, then the rest will be filled with "white"" (1,1,1)
#
# options: use hsv instead of rgb

def get_colors(img, top_n=10):
    #color, count = np.unique(np.around(np.array([colorsys.rgb_to_hsv(*x) for x in np.reshape(np.rollaxis(img,0,3),(160000,3))])/255,2), return_counts=True, axis=0)
    color, count = np.unique(np.around(np.reshape(np.rollaxis(img,0,3),(160000,3))/255,1), return_counts=True, axis=0)
    order = count.argsort()[-top_n:]
    return color[order].flatten()

def prep_color():
    path_data = expanduser('~/data/LLD-logo.hdf5')
    path_out  = expanduser("~/data/color_conditional.npz")

    #path_data = '/home/jan/Documents/uni/thesis/data/LLD-logo.hdf5'
    imgs = h5py.File(path_data, 'r')['data']

    ### multicore if imgs can be stored in memory
    with concurrent.futures.ProcessPoolExecutor() as executor:
        color_conditional = list(tqdm(executor.map(get_colors, imgs), total=len(imgs)))
    np.savez_compressed(expanduser(path_out), colors= color_conditional)

    ### non multicore
        #color_conditional = []
        #for img in tqdm(imgs):
        #    color_conditional.append(get_colors(img))


if __name__=="__main__":
    prep_color()
