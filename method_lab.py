import numpy as np
import tensorflow as tf
from deepexplain.tensorflow import DeepExplain

from experiment_file import prepare_imgs, subdir, supdir, save_result, to_extension, save_heatmap, save_array
from model_accessor import load_model, input_node, presoftmax, top_labels_and_ids


def normalize(attrs, ptile=99):
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100 - ptile)
    return np.clip(attrs / max(abs(h), abs(l)), -1.0, 1.0)


model_name = 'inception_v3'
# dir = 'methods_final_2_large'
dir = 'test_data_3'

# imgs, img_filenames = prepare_imgs('test_data_final_2_large', dir)
imgs, img_filenames = prepare_imgs(dir, dir)
# s=50
# imgs = imgs[2*s:3*s]
# img_filenames = img_filenames[2*s:3*s]

imgs = (2 * imgs / 255) - 1
tf.reset_default_graph()
cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
sess = tf.Session(config=cfg)

with DeepExplain(session=sess, graph=sess.graph) as de:
    load_model(model_name, tf.placeholder(tf.float32, [None, 224, 224, 3], 'input'), True)

    labels, ids = top_labels_and_ids(sess, imgs)

    # print(labels, ids)

    input_n = input_node()
    output_n = presoftmax()
    target_n = tf.reduce_max(output_n, 1)

    attributions = {}
    BATCH_SIZE = 10
    for i in range(0, len(imgs), BATCH_SIZE):
        print('iter %s' % i)
        batch = imgs[i: i+BATCH_SIZE]
        batch_filenames = img_filenames[i: i+BATCH_SIZE]
        # attributions['saliency'] = de.explain('saliency', target_n, input_n, batch)
        # attributions['intgrad'] = de.explain('intgrad', target_n, input_n, batch)
        # attributions['elrp'] = de.explain('elrp', target_n, input_n, batch)
        # attributions['deeplift'] = de.explain('deeplift', target_n, input_n, batch)
        attributions['occlusion'] = de.explain('occlusion', target_n, input_n, batch, window_shape=(15,15,3), step=1)


        for i, filename in enumerate(batch_filenames):
            for attr_name, attr in attributions.items():
                attr = np.mean(attr[i], axis=2)
                subdir(attr_name)
                normed = normalize(attr)
                save_heatmap(normed, to_extension(filename, 'png'))
                save_result((np.abs(normed) * 255).astype(np.uint8), to_extension(filename, 'png', '-gray'))
                save_array(attr, to_extension(filename, 'npy'))
                supdir()