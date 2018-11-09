from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io

from experiment_file import to_extension


def to_gray_level(attrs, ptile=99):
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100 - ptile)
    normed = np.clip(attrs / max(abs(h), abs(l)), -1.0, 1.0)
    return (np.abs(normed) * 255).astype(np.uint8)

root_dir = 'test_data_final_2_large'
save_dir = 'result/methods_final_2_large/random_baseline'



# for p in glob(root_dir + '/*'):
#     filename = os.path.splitext(os.path.basename(p))[0]
#     map = np.random.uniform(-1, 1, (299, 299))
#     np.save(save_dir + '/%s.npy' % filename, map)

scan_dirs = glob('result/methods_final_2_large/*')
for sub_dir in scan_dirs:
    filenames = glob(sub_dir + '/*.npy')

    for filename in filenames:
        io.imsave(sub_dir +'/' + to_extension(filename, 'png', '-gray'), to_gray_level(np.load(filename)), vmin=0, vmax=255)



