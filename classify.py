import numpy as np
from skimage import io, feature, color, filters
from skimage import exposure
import tensorflow as tf

import matplotlib.pyplot as plt

from generate_gif import generate_gif

MODEL_LOC = './InceptionModel/tensorflow_inception_graph.pb'
LABELS_LOC = './InceptionModel/imagenet_comp_graph_label_strings.txt'

graph = None
sess = None
labels = None
def load_model():
    graph = tf.Graph()
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    graph_def = tf.GraphDef.FromString(open(MODEL_LOC, 'rb').read())
    tf.import_graph_def(graph_def)

    return sess, graph

def tensor(layer):
    return graph.get_tensor_by_name('import/%s:0' % layer)

def run_network(sess, tensor, images):
    imagenet_mean = 117.0
    return sess.run(tensor, {'import/input:0': [image - imagenet_mean for image in images]})

def top_label_and_score(img):
    t_softmax = tf.reduce_mean(tensor('softmax2'), reduction_indices=0)
    scores = run_network(sess, t_softmax, [img])
    id = np.argmax(scores)

    return labels[id], scores[id]

def output_label_tensor(label):
    lab_index = np.where(np.in1d(labels, [label]))[0][0]
    t_softmax = tensor('softmax2')
    t_softmax = tf.reduce_sum(tensor('softmax2'), reduction_indices=0)
    return t_softmax[lab_index]

def output_label_tensors(label):
    lab_index = np.where(np.in1d(labels, [label]))[0][0]
    t_softmax = tensor('softmax2')
    return t_softmax[:, lab_index]

def interior_scores_from_init_img(img, init_img, label, steps=50):
    t_output = output_label_tensors(label)
    cont_images = interpolate_img(img, init_img, steps)

    outputs = run_network(sess, t_output, cont_images)
    return outputs

def integrated_gradients_from_init(img, init_img, label, steps=50):
    t_output = output_label_tensor(label)
    t_grad = tf.gradients(t_output, tensor('input'))[0]
    cont_images = interpolate_img(img, init_img, steps)

    grads = run_network(sess, t_grad, cont_images)
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
    return np.clip(attrs/ max(abs(h), abs(l)), -1.0, 1.0)

def normalize2(mask, ptile=99):
    h = np.percentile(mask, ptile)

    return np.clip(mask / h, 0, 1.0)

R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])
K = np.array([0, 0, 0])

def visualize_attrs_overlay(img, attrs, pos_ch=G, neg_ch=R, ptile=99):
    attrs = gray_scale(attrs)
    attrs = normalize(attrs, ptile)

    pos_attrs = attrs * (attrs >= 0.0)
    neg_attrs = -1 * attrs * (attrs < 0.0)
    attrs_mask = pos_attrs * pos_ch + neg_attrs * neg_ch
    vis = 0.3 * gray_scale(img) + 0.7 * attrs_mask



def visualize_attrs_windowing(img, attrs):
    attrs = np.abs(attrs)
    attrs = gray_scale(attrs)
    attrs = np.clip(attrs / np.percentile(attrs, 99), 0, 1)
    # vis = (img * attrs).astype('uint8')
    vis = (attrs * 255).astype(np.uint8)
    plt.imsave('result/exp1_ig.png', vis)
    plt.imshow(vis)
    plt.show()

def generate_init_img(shape=(224, 224, 3)):
    # return np.random.randint(0, 256, shape, dtype='uint8')
    # return np.full(shape=shape, fill_value=1, dtype='uint8') * B
    return np.zeros(shape=shape, dtype='uint8')

def build_masking_graph(var_init=None):
    with tf.Graph().as_default() as graph:
        if var_init == None:
            mask = tf.Variable(tf.truncated_normal((224, 224, 1), 4, 0.2))
        else:
            mask = tf.Variable(tf.constant(var_init))
        input = tf.placeholder(tf.float32, [1, 224, 224, 3], name='m_input')

        sig_mask_op = tf.sigmoid(mask, name='mask_sigmoid')
        masked_input = tf.multiply(input, sig_mask_op) - tf.constant(117.0)


    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    graph_def = tf.GraphDef.FromString(open(MODEL_LOC, 'rb').read())
    tf.import_graph_def(graph_def, input_map={'input:0': masked_input})

    return  sess, graph, mask,  sig_mask_op

def masking_graph_cost(label, sig_mask_op):
    mask_mean = tf.reduce_mean(sig_mask_op)
    # mid_var = tf.reduce_mean(tf.square(sig_mask_op - 0.5))
    # mid_diff = tf.square(tf.reduce_mean(sig_mask_op - 0.5))
    target_softmax = output_label_tensor(label)
    max_min_control = tf.placeholder(tf.float32, name='max_min_control')
    # return 0.1 * (1.0 - target_softmax) + 0.3 * (-mid_var + mid_diff) - 0.3 * mask_mean, target_softmax
    return (1.0 - target_softmax) + max_min_control * mask_mean, target_softmax

def save_progress(records, i, norm_masked_img):
    records[i] = np.clip(norm_masked_img * 255, 0, 255).astype(np.uint8)

directory = 'result/near_one_init'
label = 'indigo bunting'
img = io.imread('%s.jpg' % label)
labels = np.array(open(LABELS_LOC).read().split("\n"))
sess, graph = load_model()


masked = np.load('masked.npy').squeeze()

plt.imshow(masked)
plt.show()
top_label, score = top_label_and_score(masked)
print("Top label: %s, score: %f" % (top_label, score))


