import os
from random import sample

import numpy as np
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt
from generate_gif import generate_gif
import tensorflow as tf
save_directory = 'result'

class Record:

    def __init__(self, frame_num):
        self.masked = np.zeros((frame_num, 224, 224, 3), dtype=np.float32)
        self.mask = np.zeros((frame_num, 224, 224), dtype=np.float32)
        self.dir = save_directory
        if not os.path.exists(self.dir + '/feat_progress'):
            # os.makedirs(self.dir + '/mask_progress')
            os.makedirs(self.dir + '/masked_progress')
            # os.makedirs(self.dir + '/noise_progress')
            # os.makedirs(self.dir + '/feat_progress')

        self.progress = 0

    def save_progress(self, masked, mask):
        self.masked[self.progress] = masked
        self.mask[self.progress] = mask
        self.progress += 1

    def save_img_progress(self, masked, mask, noise, feat):
        io.imsave(self.dir + '/masked_progress/%s.png' % self.progress, to_img_type255(masked))
        #io.imsave(self.dir + '/mask_progress/%s.png' % self.progress, to_img_type(mask), cmap='gray')
        #io.imsave(self.dir + '/noise_progress/%s.png' % self.progress, to_img_type255(noise))

        # feat = feat[1:].reshape(20, -1)
        # feat = np.repeat(feat, 10, axis=1)
        # feat = np.repeat(feat, 10, axis=0)
        # plt.imsave(self.dir + '/feat_progress/%s.png' % self.progress, feat, cmap='OrRd')

        self.progress += 1

    def write_records(self):
        progress_end = self.progress - 1
        # generate_gif(self.dir + '/masked.gif', to_img_type(self.masked))
        generate_gif(self.dir + '/mask.gif', to_img_type(self.mask[:progress_end]))
        # generate_3d_gif(directory + '/3d_mask_of_%s.gif' % label, mask_records)

        io.imsave(self.dir + '/feature.png', to_img_type(self.masked[progress_end]), vmin=0, vmax=255)
        io.imsave(self.dir + '/feature_mask.png', to_img_type(self.mask[progress_end]), cmap='gray', vmin=0, vmax=255)
        np.save(self.dir + '/feature_mask.npy', self.mask[progress_end])
        # np.save(self.dir + '/mask_progress.npy', self.mask)
        print('save at: %s' % self.dir)

    def save_checkpoint(self):
        true_progress = self.progress - 1
        checkpoint_dir = self.dir + '/checkpoint_%s' % true_progress
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('checkpoint at frame %s' % true_progress)
        generate_gif(checkpoint_dir + '/mask.gif', to_img_type(self.mask[:self.progress]))
        io.imsave(checkpoint_dir + '/feature_mask.png', to_img_type(self.mask[true_progress]), cmap='gray', vmin=0, vmax=255)




def prepare_img(label, id, save_dir, load_dir='test_data'):

    label_and_id = label if id == 0 else label + str(id)

    if save_dir:
        global save_directory
        save_directory = os.path.join(save_directory, save_dir, label_and_id)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    img = ndimage.imread('%s/%s.jpg' % (load_dir, label_and_id), mode='RGB')

    return img.astype(np.float32)

def get_filenames(dir, filename='*'):
    filenames = list(tf.gfile.Glob(os.path.join(dir, filename)))
    filenames.sort()
    return filenames

def prepare_imgs(imgs_dir, save_dir, limit=None):
    if save_dir:
        global save_directory
        save_directory = os.path.join(save_directory, save_dir)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    imgs = []
    filenames = get_filenames(imgs_dir)
    if limit:
        filenames = sample(filenames, limit)

    for filename in filenames:
        imgs.append(ndimage.imread(filename, mode='RGB'))

    return np.array(imgs).astype(np.float32), filenames

def to_img_type255(array):
    return np.clip(array, 0, 255).astype(np.uint8)

def to_img_type(array):
    return np.clip(array * 255, 0, 255).astype(np.uint8)

def save_result(img, filename='feature.png'):
    io.imsave(save_directory + '/' + filename, img, vmin=0, vmax=255)

def save_heatmap(heat_map, filename='heatmap.png'):
    abs_max = np.max(np.abs(heat_map))
    plt.imsave(save_directory + '/' + filename, heat_map, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)

def save_array(array, filename):
    np.save(save_directory + '/' + filename, array)

def subdir(dir_name):
    global save_directory
    full_subdir = os.path.join(save_directory, dir_name)
    if not os.path.exists(full_subdir):
        os.makedirs(full_subdir)

    save_directory = full_subdir

def supdir():
    global save_directory
    save_directory = os.path.dirname(save_directory)


def to_extension(path_to_file, ext, postfix=''):
    return '%s%s.%s' % (os.path.splitext(os.path.basename(path_to_file))[0], postfix, ext)

if __name__ == '__main__':
    save_dir = 'test_data_final_2_large'
    filenames = get_filenames('test_data_nips')
    filenames = sample(filenames, 200)


    if os.path.exists(save_dir):
        tf.gfile.DeleteRecursively(save_dir)
    os.makedirs(save_dir)



    # sess = tf.Session()
    # input_img = tf.placeholder(tf.uint8, [1, None, None, 3])
    # img_shape = tf.shape(input_img)[1:3]
    # rect_length = tf.maximum(img_shape[0], img_shape[1])
    # rect_length = tf.minimum(img_shape[0], img_shape[1])

    # cropped = tf.image.resize_image_with_crop_or_pad(input_img, rect_length, rect_length)
    # processed_img = tf.image.resize_bilinear(cropped, [299, 299], True)


    for filename in filenames:
        img = ndimage.imread(filename, mode='RGB')

        # result = sess.run(processed_img, {input_img: [img]}).astype(np.uint8)

        # io.imsave('%s/%s' % (save_dir, to_extension(filename, 'png')), result[0])
        io.imsave('%s/%s' % (save_dir, to_extension(filename, 'png')), img)

