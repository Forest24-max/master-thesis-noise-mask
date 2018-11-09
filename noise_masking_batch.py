from datetime import datetime
from glob import glob

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from tensorflow.contrib.opt import LazyAdamOptimizer

import experiment_file
from experiment_file import save_heatmap, Record, prepare_imgs, to_extension, save_array, save_result, \
    to_img_type
from model_accessor import load_model, presoftmax

graph = None
sess = None
labels = None

img_size = 299

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


            noise_background = tf.random_uniform([1, img_size, img_size, 3], 0, 255)
            sig_mask_op = tf.sigmoid(mask_var, name='mask_sigmoid')
            masked_input = tf.multiply(input, sig_mask_op) + tf.multiply(noise_background, 1 - sig_mask_op)


    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    load_model(model_name, masked_input)

    return sess, graph, mask_var, sig_mask_op, masked_input


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

        feat_map = presoftmax()
        truth_feat_map = tf.placeholder(dtype=tf.float32, name='target_feat_map')
        weight = tf.placeholder(dtype=tf.float32, name='loss_weight')

        alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        beta = tf.placeholder(dtype=tf.float32, name='beta')


        feat_map_diff = (truth_feat_map - feat_map)
        feat_map_loss = tf.reduce_mean(tf.square(feat_map_diff) * weight, name='feat_loss')


        # target_softmax = output_label_tensor(label)
        feat_map_loss = alpha * feat_map_loss
        # total_loss = tf.add(feat_map_loss, total_var(sig_mask_op)*0.5 + beta * mask_mean, name='total_loss')
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


class Iteration:
    def __init__(self, max, log, checkpoint=np.inf):
        self.max = max
        self.log = log
        self.checkpoint = checkpoint
        self.i = 0

    @property
    def log_size(self):
        return self.max // self.log + 1

imgs, filenames = prepare_imgs('test_data_4', 'test_data_4_25')
model_name = 'inception_v3'
beta = 1
learning_rate = 0.01
iteration_num = 50000

checkpoint_iter = 50000


sess, graph, mask_var, sig_mask_op, masked_input = build_masking_graph(model_name, 4)

# list_tensors()

cost_op, last_feat_map_op, loss_terms = masking_graph_cost(sig_mask_op)

optimizer = LazyAdamOptimizer(learning_rate)
opt_op = optimizer.minimize(cost_op, var_list=[mask_var])


iter = Iteration(max=iteration_num, log=50, checkpoint=checkpoint_iter)


# tensorboard
# loss_terms_placeholder = tf.placeholder(tf.float32)
# tf.summary.scalar('loss_terms', loss_terms_placeholder)
# writers = tensorboard_writers(experiment_file.save_directory, loss_terms)
# merged_summary = tf.summary.merge_all()
#

AM_LOSS_THRESHOLD = 1
MASK_CONVERGENCE_THRESHOLD = 10
def run_optimization(img_i, img, others_weight, iter):
    i = 0

    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    target_feat_map = sess.run(last_feat_map_op, {'input/mask_var:0': np.full([1, img_size, img_size, 1], 4), 'input/m_input:0':[img]})
    full_noise_feat_map = sess.run(last_feat_map_op, {'input/mask_var:0': np.full([1, img_size, img_size, 1], -6), 'input/m_input:0':[img]})

    alpha = 1 / np.mean(np.square(target_feat_map - full_noise_feat_map))


    loss_weight = np.full_like(target_feat_map, others_weight)
    loss_weight[:, np.argmax(target_feat_map, axis=1)] = 1

    last_log_mask_sum = np.inf
    # accumulation = np.zeros((img_size, img_size))
    while i <= iter.max:

        *losses, _ = sess.run([*loss_terms, opt_op], {'input/m_input:0': [img],
                                                      'loss/target_feat_map:0': target_feat_map,
                                                      'loss/loss_weight:0': loss_weight,
                                                      'loss/alpha:0': alpha,
                                                       'loss/beta:0': beta})

        if losses[0] > AM_LOSS_THRESHOLD:
            print('exceed threshold')
            break

        if i % iter.log == 0 and 0 < i < iter.max:
            sig_mask, masked_input_val = \
                sess.run([sig_mask_op, masked_input], {'input/m_input:0': [img]})


            sig_mask = sig_mask.squeeze()# normalize2(sig_mask)
            # accumulation += sig_mask

            mask_sum = np.sum(sig_mask)
            print('img-%s iter: %s max_mask: %s mask_sum: %s' % (img_i, i, np.max(sig_mask), mask_sum))

            if last_log_mask_sum - mask_sum <= MASK_CONVERGENCE_THRESHOLD:
                return sig_mask

            last_log_mask_sum = mask_sum

        i += 1

    sig_mask, masked_input_val = sess.run([sig_mask_op, masked_input],
                                          {'input/m_input:0': [img]})
    sig_mask = sig_mask.squeeze()
    # accumulation += sig_mask

    return sig_mask

for i, (img, filename) in enumerate(zip(imgs, filenames), start=1):
    print('progress %s' % i)
    result = run_optimization(i, img, 0.25, iter)
    # record.write_records()

    save_heatmap(result, to_extension(filename, 'png'))
    save_array(result, to_extension(filename, 'npy'))
    result = normalize2(result)
    save_result(to_img_type(result), to_extension(filename, 'png', '-gray'))
