from glob import glob

import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.opt import LazyAdamOptimizer
from scipy.stats import norm

from experiment_file import prepare_img, save_heatmap, Record
from model_accessor import load_model, last_volume, output_label_tensor, presoftmax, mid_volume, output_node

graph = None
sess = None
labels = None

img_size = 224

def normalize2(mask, ptile=99.9):
    h = np.percentile(mask, ptile)
    return np.clip(mask / h, 0, 1.0)

def gaussian_filter(size, sigma=1.0, channels=3):
    interval = size // 2
    x = np.linspace(-interval, interval, size)
    kern1d = norm.pdf(x, scale=sigma)
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    np_filter = np.array(kernel, dtype = np.float32)
    np_filter = np_filter.reshape((size, size, 1, 1))
    np_filter = np.repeat(np_filter, channels, axis = 2) #[size, size, channel, 1]
    return tf.constant(np_filter)


def build_masking_graph(model_name, var_init=None):


    with tf.Graph().as_default() as graph:
        with tf.name_scope('input'):
            input = tf.placeholder(tf.float32, [1, img_size, img_size, 3], name='m_input')


            if var_init == None:
                mask_var = tf.Variable(tf.truncated_normal((1, img_size, img_size, 1), 4, 0.1), name='mask_var')
            else:
                mask_var = tf.Variable(tf.constant(var_init, dtype=tf.float32, shape=(1, img_size, img_size, 1)), name='mask_var')




            # noise_r = tf.truncated_normal([1, img_size, img_size, 1], 122.46, 70.63)
            # noise_g = tf.truncated_normal([1, img_size, img_size, 1], 114.26, 68.61)
            # noise_b = tf.truncated_normal([1, img_size, img_size, 1], 101.37, 71.93)
            # noise_background = tf.clip_by_value(tf.concat([noise_r, noise_g, noise_b], axis=3), 0, 255)
            noise_background = tf.random_uniform([1, img_size, img_size, 3], 0, 255)


            sig_mask_op = tf.sigmoid(mask_var, name='mask_sigmoid')

            pool_octave = tf.placeholder(tf.int32, shape=[], name='pool_octave')

            # mask_pooled_ops = tf.stack([make_shrink_and_expand(2 ** s, sig_mask_op, img_size) for s in range(5, -1, -1)])
            # sig_mask_op = mask_pooled_ops[pool_octave]

            masked_input = tf.multiply(input, sig_mask_op) + tf.multiply(noise_background, 1 - sig_mask_op)


    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    load_model(model_name, masked_input)

    return sess, graph, mask_var, sig_mask_op, masked_input, noise_background

def make_shrink_and_expand(size, sig_mask_op, img_size):
    if size == 1:
        return sig_mask_op
    return tf.image.resize_bilinear(tf.nn.avg_pool(sig_mask_op, [1, size, size, 1], [1, size, size, 1], 'SAME'), tf.constant([img_size, img_size]))

def total_var(sig_mask_op):
    pad1 = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0]])
    pad2 = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0]])
    pixel_dif1 = tf.pad(sig_mask_op[:, 1:, :, :] - sig_mask_op[:, :-1, :, :], pad1)
    pixel_dif2 = tf.pad(sig_mask_op[:, :, 1:, :] - sig_mask_op[:, :, :-1, :], pad2)

    tv = tf.reduce_mean(tf.abs(pixel_dif1) + tf.abs(pixel_dif2))
    return tv

def masking_graph_cost(sig_mask_op):
    with tf.name_scope('loss'):
        mask_mean = tf.reduce_mean(sig_mask_op, name='mask_loss')

        # feat_map = mid_volume()
        # feat_map = last_volume()
        feat_map = presoftmax()
        truth_feat_map = tf.placeholder(dtype=tf.float32, name='target_feat_map')


        alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        beta = tf.placeholder(dtype=tf.float32, name='beta')


        feat_map_diff = (truth_feat_map - feat_map)

        feat_map_loss = tf.reduce_mean(tf.square(feat_map_diff), name='feat_loss')


        feat_map_loss = alpha * feat_map_loss

        total_loss = tf.add(feat_map_loss, beta * mask_mean, name='total_loss')
    return total_loss, feat_map, (feat_map_loss, mask_mean)



def tensorboard_writers(root_dir, loss_terms):

    delete_board_directories(root_dir)
    id = datetime.now().microsecond
    writers = [tf.summary.FileWriter(root_dir + '/%s-%s' % (loss_terms[0].name.replace('/', '-'), id), sess.graph)]
    for term in loss_terms[1:]:
        writers.append(tf.summary.FileWriter(root_dir + '/%s-%s' % (term.name.replace('/', '-'), id)))
    return writers

def delete_board_directories(root_dir):
    del_dirs = glob(root_dir + '/loss-*')

    for dir in del_dirs:
            tf.gfile.DeleteRecursively(dir)


def save_dir_name(model_name, scheme_name, beta, learning_rate, iteration_num):
    return model_name + ('/%s l-%s lr-%s iter-%s' % (scheme_name, beta, str(learning_rate).replace('.', ''), iteration_num))

def compute_mask_shape(octave):
    return [224 // (2 ** octave), 224 // (2 ** octave)]

class Iteration:
    def __init__(self, max, log, checkpoint=np.inf):
        self.max = max
        self.log = log
        self.checkpoint = checkpoint
        self.i = 0

    @property
    def log_size(self):
        return self.max // self.log + 1

img_name = 'cricket'
id =2
method = 'nsb'
model_name = 'vgg_16'
beta = 1
learning_rate = 0.012
iteration_num = 40000

checkpoint_iter = 10000

img = prepare_img(img_name, id, save_dir_name(model_name, method, beta, learning_rate, iteration_num), 'test_data')
sess, graph, mask_var, sig_mask_op, masked_input, noise_bg_op = build_masking_graph(model_name, 4)
# list_tensors()
# beta=1.1
# learning_rate=0.01
# iteration_num = 50000
cost_op, last_feat_map_op, loss_terms = masking_graph_cost(sig_mask_op)

optimizer = LazyAdamOptimizer(learning_rate)
opt_op = optimizer.minimize(cost_op, var_list=[mask_var])


iter = Iteration(max=iteration_num, log=100, checkpoint=checkpoint_iter)
record = Record(iter.log_size)

AM_LOSS_THRESHOLD = 1

def run_optimization(record, iter):
    i = 0
    pool_octave = 5
    beta_tmp = beta

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    target_feat_map = sess.run(last_feat_map_op, {'input/mask_var:0': np.full([1, img_size, img_size, 1], 4), 'input/pool_octave:0': pool_octave, 'input/m_input:0':[img]})
    full_noise_feat_map = sess.run(last_feat_map_op, {'input/mask_var:0': np.full([1, img_size, img_size, 1], -6), 'input/pool_octave:0': pool_octave, 'input/m_input:0':[img]})


    alpha = 1 / np.mean(np.square(target_feat_map - full_noise_feat_map))
    print(alpha, 1 / alpha)

    loss_terms_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss_terms', loss_terms_placeholder)

    writers = tensorboard_writers(record.dir, loss_terms)
    merged_summary = tf.summary.merge_all()


    while i <= iter.max:
        # if i <= 450:
        #     beta_tmp = 0.2 + (beta - 0.2) * i / 450
        #     if i == 0:
        #         pool_octave = 0
        #
        #     elif i == 100:
        #         pool_octave = 1
        #
        #     elif i == 150:
        #         pool_octave = 2
        #
        #     elif i == 200:
        #         pool_octave = 3
        #
        #     elif i == 300:
        #         pool_octave = 4
        #
        #     elif i == 400:
        #         pool_octave = 5


        *losses, _ = sess.run([*loss_terms, opt_op], {'input/pool_octave:0': pool_octave,
                                                      'input/m_input:0': [img],
                                                      'loss/target_feat_map:0': target_feat_map,
                                                      #'loss/loss_weight:0': loss_weight,
                                                      'loss/alpha:0': alpha,
                                                       'loss/beta:0': beta_tmp})

        if losses[0] > AM_LOSS_THRESHOLD:
            print('exceed threshold')
            sig_mask, masked_input_val = sess.run([sig_mask_op, masked_input],
                                                  {'input/pool_octave:0': pool_octave, 'input/m_input:0': [img]})
            sig_mask = sig_mask.squeeze()
            norm_masked = normalize2(masked_input_val[0])
            record.save_progress(norm_masked, sig_mask)
            break

        for loss, writer in zip(losses, writers):
            summary = sess.run(merged_summary, {loss_terms_placeholder: loss})
            writer.add_summary(summary, i)


        if i % iter.log == 0:
            sig_mask, masked_input_val, feat_map = \
                sess.run([sig_mask_op, masked_input, last_feat_map_op], {'input/pool_octave:0': pool_octave,
                                                                         'input/m_input:0': [img]})


            sig_mask = sig_mask.squeeze()# normalize2(sig_mask)
            masked = masked_input_val.squeeze()
            noise_bg = masked
            # noise_bg = noise_bg.squeeze()
            feat_map = feat_map.squeeze()
            norm_masked = normalize2(masked_input_val.squeeze())
            # record.save_progress(norm_masked, sig_mask)

            record.save_img_progress(masked, sig_mask, noise_bg, feat_map)
            max_neuron_loss = np.max(np.abs(target_feat_map - feat_map))
            print('iter: %s n_loss: %s max_mask: %s mask_sum: %s' % (i, max_neuron_loss, np.max(sig_mask), np.sum(sig_mask)))

        if 0 < i < iter.max and i % iter.checkpoint == 0:
            record.save_checkpoint()

        i += 1


    return sig_mask

heatmap = run_optimization(record, iter)

np.save('mask.npy', sess.run(sig_mask_op))

record.write_records()
save_heatmap(heatmap)
