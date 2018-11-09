import glob as glob
import os

import numpy as np
from scipy import ndimage
from skimage import color
from skimage.measure import compare_ssim as ssim

from experiment_file import get_filenames, prepare_imgs


def load_saliency_maps(dir):
    filenames = get_filenames(dir, '*-gray.png')
    return [ndimage.imread(filename, mode='L') for filename in filenames]



def batch_ssim(input_imgs, maps):
    ssims = []
    for img, map in zip(input_imgs, maps):
        ssims.append(ssim(img, map))

    return ssims


root_dir = 'result/methods_final_2_large'
input_img_dir = 'test_data_final_2_large'

input_imgs, _ = prepare_imgs(input_img_dir, None)
input_imgs = np.array([color.rgb2gray(img.astype(np.uint8)) for img in input_imgs]).astype(np.uint8)

average_ssim = {}

for sub_dir in glob.glob(root_dir + '/*'):
    method_name = os.path.basename(sub_dir)

    maps = load_saliency_maps(sub_dir)
    ssims = batch_ssim(input_imgs, maps)
    average_ssim[method_name] = (np.mean(ssims), np.std(ssims))




for method_name, avg_ssim in average_ssim.items():
    print('%s: mean: %s std: %s' % (method_name, avg_ssim[0], avg_ssim[1]))