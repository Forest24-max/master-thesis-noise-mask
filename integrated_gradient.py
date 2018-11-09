import numpy as np
from skimage import io, feature, color, filters
from skimage import exposure
import tensorflow as tf

import matplotlib.pyplot as plt

from experiment_file import prepare_img, save_result, save_heatmap
from generate_gif import generate_gif
from model_accessor import output_label_tensors, output_label_tensor, load_model, run_network, top_label_and_score, \
    input_node


graph = None
sess = None
labels = None

STEPS = 100
BATCH_SIZE = 10


def interior_scores_from_init_img(img, init_img, label, steps=STEPS):
    t_output = output_label_tensors(label)
    cont_images = interpolate_img(img, init_img, steps)

    outputs = run_network(sess, t_output, cont_images)
    return outputs

def integrated_gradients_from_init(img, init_img, label, steps=STEPS):
    t_output = output_label_tensor(label)
    t_grad = tf.gradients(t_output, input_node())[0]
    cont_images = interpolate_img(img, init_img, steps)

    grads = np.zeros_like(cont_images, np.float32)
    for i in range(0, steps, BATCH_SIZE):
        grads[i: i + BATCH_SIZE] = run_network(sess, t_grad, cont_images[i: i + BATCH_SIZE])
    #note positive/negative
    return img * np.average(grads, axis=0), np.average(np.abs(grads), axis=(1,2,3))

def interpolate_img(img, init_img, steps):
    img_diff = img - init_img
    c_imgs = [(init_img + img_diff * float(i) / steps).astype(np.uint8) for i in range(1, steps + 1)]
    c_imgs.append(img)
    return c_imgs

def gray_scale(attr):
    attr = np.average(attr, axis=2)
    return np.tile(attr[:, :, np.newaxis], [1, 1, 3])

def normalize(attrs, ptile=99):
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100 - ptile)
    return np.clip(attrs / max(abs(h), abs(l)), -1.0, 1.0)

R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
def visualize_attrs_overlay(img, attrs, pos_ch=G, neg_ch=R, ptile=99):
    attrs = gray_scale(attrs)
    attrs = normalize(attrs, ptile)

    pos_attrs = attrs * (attrs >= 0.0)
    neg_attrs = -1 * attrs * (attrs < 0.0)
    attrs_mask = pos_attrs * pos_ch + neg_attrs * neg_ch
    vis = 0.3 * gray_scale(img) + 0.7 * attrs_mask

    return vis

def visualize_attrs_windowing(img, attrs):
    attrs = np.abs(attrs)
    attrs = gray_scale(attrs)
    attrs = np.clip(attrs / np.percentile(attrs, 99), 0, 1)
    # vis = (img * attrs).astype('uint8')
    vis = (attrs * 255).astype(np.uint8)
    return vis

def generate_init_img(shape=(224, 224, 3)):
    # return np.random.randint(0, 256, shape, dtype='uint8')
    return np.full(shape=shape, fill_value=0, dtype='uint8')
    # return np.zeros(shape=shape, dtype='uint8')



label = 'cricket'
id = 2
model_name = 'vgg_16'
save_dir = 'ig/' + model_name

img = prepare_img(label, id, save_dir, load_dir='test_data')

graph = tf.Graph()
graph.as_default()
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.InteractiveSession(graph=graph, config=cfg)
load_model(model_name, tf.placeholder(tf.float32, [None, 224, 224, 3], 'input'))


t_label, score = top_label_and_score(sess, img)

print('label: %s score: %s' % (t_label, score))
# init_img = generate_init_img()
# intgrads_attrs, avg_grads = integrated_gradients_from_init(img, init_img, t_label, steps=STEPS)
# vis = visualize_attrs_windowing(img, intgrads_attrs)
# vis = normalize(np.average(intgrads_attrs, axis=2))
# save_heatmap(vis)


# plt.plot(avg_grads)
# plt.show()
# int_scores = interior_scores_from_init_img(img, init_img, label, steps=100)
# plt.plot(int_scores)
# plt.show()

# frames = np.array(interpolate_img(img, init_img, steps=100))
# generate_gif('result/rand_init_%s.gif' % label, frames)
