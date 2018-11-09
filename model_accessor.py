import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import inception, vgg

import os

from datasets import imagenet

MODEL_GRAPH_DIR = 'models'
LABELS_LOC = './models/imagenet_labels.txt'

_model_name = None
labels = None


LAST_VOLUME_OP = {'resnet_v1': 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu',
                'vgg_16': 'vgg_16/pool5/MaxPool',
                'inception_v1': 'InceptionV1/InceptionV1/Mixed_5c/concat',
                'inception_v3': 'InceptionV3/InceptionV3/Mixed_7c/concat'}

MID_VOLUME_OP = {
'inception_v3': 'InceptionV3/InceptionV3/Mixed_7b/concat',
'inception_v1': 'InceptionV1/InceptionV1/Mixed_4e/concat',
'vgg_16': 'vgg_16/conv4/conv4_3/Relu',
'resnet_v1': 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu'}

PRE_SOFTMAX_OP = {'resnet_v1': 'resnet_v1_50/SpatialSqueeze',
                'vgg_16': 'vgg_16/fc8/squeezed',
                'inception_v1': 'InceptionV1/Logits/SpatialSqueeze',
                'inception_v3': 'InceptionV3/Logits/SpatialSqueeze'}

OUTPUT_OP = {'resnet_v1': 'resnet_v1_50/predictions/Reshape_1',
                'vgg_16': 'vgg_16/fc8/squeezed',
                'inception_v1': 'InceptionV1/Logits/Predictions/Reshape_1',
                'inception_v3': 'InceptionV3/Predictions/Reshape_1'}


nodes = {}

def load_model(model_name, input_imgs, preprocessed=False):
    global _model_name, labels, nodes
    _model_name = model_name

    processed_imgs = input_imgs if preprocessed else PREPROCESSORS[model_name](input_imgs)

    graph_def = tf.GraphDef.FromString(open('models/%s.pb' % model_name, 'rb').read())
    tf.import_graph_def(graph_def, input_map={'input:0': processed_imgs})

    #list_tensors()
    nodes['Input'] = input_imgs
    nodes['Mid Volume'] = tensor(MID_VOLUME_OP[model_name])
    nodes['Last Volume'] = tensor(LAST_VOLUME_OP[model_name])
    nodes['Presoftmax'] = tensor(PRE_SOFTMAX_OP[model_name])
    nodes['Output'] = tensor(OUTPUT_OP[model_name])


    labels = open(LABELS_LOC).read().split("\n")
    if not model_name.startswith('inception'):
        labels = labels[1:]

    labels = np.array(labels)


def to_list(table):
    list = [None] * len(table)
    for k, v in table.items():
        list[k] = v
    return list

def tensor(layer_name):
    return tf.get_default_graph().get_tensor_by_name('import/%s:0' % layer_name)



def inception_preprocess_images(images):
    images = tf.divide(images, 255.0)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)
    return images


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
COLOR_MEAN_ARRAY = np.array([_R_MEAN, _G_MEAN, _B_MEAN]).reshape((1, 1, 1, -1))
def vgg_preprocess_images(images):
    mean = tf.constant(COLOR_MEAN_ARRAY, tf.float32)
    return images - mean

def presoftmax():
    return nodes['Presoftmax']

def last_volume():
    return nodes['Last Volume']

def mid_volume():
    return nodes['Mid Volume']

def input_node():
    return nodes['Input']

def output_node():
    return nodes['Output']

def output_label_tensor(label):
    lab_index = np.where(np.in1d(labels, [label]))[0][0]

    return output_id_tensor(lab_index)

def output_id_tensor(id):
    t_softmax = output_node()[0]
    return t_softmax[id]

def presoftmax_label_tensor(label):
    lab_index = np.where(np.in1d(labels, [label]))[0][0]
    t_presoftmax = presoftmax()[0]

    return t_presoftmax[lab_index]

def output_label_tensors(label):
    lab_index = np.where(np.in1d(labels, [label]))[0][0]
    t_softmax = output_node()
    return t_softmax[:, lab_index]

def top_label_and_score(sess, img):
    t_softmax = output_node()[0]
    scores = run_network(sess, t_softmax, [img])
    id = np.argmax(scores)

    return labels[id], scores[id]

def top_labels_and_ids(sess, imgs):
    scores = run_network(sess, output_node(), imgs)
    ids = np.argmax(scores, axis=1)
    return labels[ids], ids

def run_network(sess, tensor, images):
    return sess.run(tensor, {input_node(): images})


def list_tensors():
    for op in tf.get_default_graph().get_operations():
        print(op.values())


PREPROCESSORS = {
'inception_v1': inception_preprocess_images,
'inception_v3': inception_preprocess_images,
'vgg_16': vgg_preprocess_images,
'resnet_v1': vgg_preprocess_images
}


if __name__ == '__main__':

    sess = tf.InteractiveSession()
    input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='m_input')
    load_model('vgg_16', input)

