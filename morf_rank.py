import numpy as np
import tensorflow as tf
import glob as glob
import os
from experiment_file import get_filenames, prepare_imgs, save_array
import matplotlib.pyplot as plt

from model_accessor import load_model, top_labels_and_ids, output_node, run_network, output_id_tensor, presoftmax


def load_saliency_maps(dir):
    filenames = get_filenames(dir, '*.npy')
    return [np.load(filename) for filename in filenames]


def occlude_patch(input_img, occulude_locs, size=3, replacement=0):
    stride = size // 2
    max_loc = input_img.shape[:2]
    occulude_locs = np.transpose(occulude_locs)
    for occulude_loc in occulude_locs:
        patch_range = [occulude_loc - stride, occulude_loc + stride + 1]

        patch_range = np.clip(patch_range, 0, max_loc)
        #
        patch_shape = patch_range[1] - patch_range[0]
        patch_shape = np.append(patch_shape, 3)

        replacement = np.random.uniform(0, 255, patch_shape)
        input_img[patch_range[0][0]: patch_range[1][0], patch_range[0][1]: patch_range[1][1]] = replacement

    return input_img

def occlude_pixel(input_img, occlude_locs, replacement=0):
    # replacement = np.random.uniform(0, 255, (len(occlude_locs[0]), 3))
    input_img[occlude_locs[0], occlude_locs[1]] = replacement
    return input_img

def run_morf(sess, target_id, saliency_map, input_img,
             occulude_method=occlude_pixel,
             occlusion_iter=800,
             occlusion_batch=100):

    input_img = np.copy(input_img)
    pixel_ranking = np.unravel_index(np.argsort(saliency_map, axis=None)[::-1], saliency_map.shape)
    pixel_ranking = np.array(pixel_ranking)

    scores = [eval_img(sess, input_img, target_id)]

    for i in range(0, occlusion_iter * occlusion_batch, occlusion_batch):
        input_img = occulude_method(input_img, pixel_ranking[:, i: i + occlusion_batch])
        # if i % 1000 == 0:
           # plt.imsave('result/morf/%s/%s-%s.png' % (method_name, target_id, i), input_img.astype(np.uint8))

        scores.append(eval_img(sess, input_img, target_id))

    # plt.imsave('result/morf/%s/%s.png' % (method_name, target_id), input_img.astype(np.uint8))
    return np.array(scores)

def eval_img(sess, img, target_id):
    return run_network(sess, presoftmax(), [img])[0][target_id]

def record_morfs(method_name, input_imgs, img_ids, maps, avg_records):
    print('evaluating %s' % method_name)
    all_scores = []

    for map, input_img, id in zip(maps, input_imgs, img_ids):
        score_drops = run_morf(sess, id, map, input_img)
        all_scores.append(score_drops)

    all_scores = np.array(all_scores)
    avg_records[method_name] = np.mean(all_scores - all_scores[:, 0].reshape(-1, 1), axis=0)

    save_array(all_scores, method_name + '.npy')


def plot_curve(curve_of_methods):
    for method_name, curve in curve_of_methods.items():
        plt.plot(curve, label=method_name)

    plt.legend()
    plt.show()

def plot_diff_curve(curve_of_methods):
    for method_name, curve in curve_of_methods.items():
        plt.plot(curve[:-1] - curve[1:], label=method_name)

    plt.legend()
    plt.show()

def calculate_aopc(curve_records):
    for method_name, curve in curve_records.items():
        aopc = np.mean(curve)
        print('%s: %s' % (method_name, aopc))


root_dir = 'result/methods_final_2_large'
input_img_dir = 'test_data_final_2_large'
# root_dir = 'result/test_data_3'
# input_img_dir = 'test_data_3'
input_imgs, _ = prepare_imgs(input_img_dir, 'morf')


cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.Session(config=cfg)
input_ph = tf.placeholder(tf.float32, [None, 299, 299, 3], name='m_input')
load_model('inception_v3', input_ph)

_, ids = top_labels_and_ids(sess, input_imgs[:100])
_, ids2 = top_labels_and_ids(sess, input_imgs[100:])
ids = np.append(ids, ids2)
# _, ids = top_labels_and_ids(sess, input_imgs)

average_score_drop_of_methods = {}
for sub_dir in glob.glob(root_dir + '/*'):
    method_name = os.path.basename(sub_dir)
    # if method_name != 'occlusion':
    #    continue

    maps = load_saliency_maps(sub_dir)

    record_morfs(method_name, input_imgs, ids, maps, average_score_drop_of_methods)




plot_curve(average_score_drop_of_methods)
# plot_diff_curve(average_score_drop_of_methods)
# calculate_aopc(average_score_drop_of_methods)