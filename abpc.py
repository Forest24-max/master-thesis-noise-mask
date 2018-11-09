import numpy as np
import tensorflow as tf
import glob as glob
import os
from experiment_file import get_filenames, prepare_imgs, save_array
import matplotlib.pyplot as plt

from model_accessor import load_model, top_labels_and_ids, output_node, run_network, output_id_tensor, presoftmax


def load_scores(dir):
    filenames = get_filenames(dir, '*.npy')
    return {os.path.splitext(os.path.basename(filename))[0]: np.load(filename) for filename in filenames}


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

def calculate_aoc(curve_records):
    for method_name, curve in curve_records.items():
        aoc = np.sum(curve)
        print('%s: %s' % (method_name, aoc))

dir_name = 'pixel-r-presoftmax-200l-bch100-800'
morf_dir = 'result/morf-' + dir_name
lerf_dir = 'result/lerf-' + dir_name

morf_scores = load_scores(morf_dir)
lerf_scores = load_scores(lerf_dir)
# scores['random_baseline']
avg_gap = {}
# aopc_std = {}
for method_name in morf_scores:
    morf_avg_score = np.mean(morf_scores[method_name] - morf_scores[method_name][:, 0].reshape(-1, 1), axis=0)
    lerf_avg_score = np.mean(lerf_scores[method_name] - lerf_scores[method_name][:, 0].reshape(-1, 1), axis=0)

    avg_gap[method_name] = lerf_avg_score - morf_avg_score
    # aopc_std[method_name] = np.std(np.mean(record[:, 0].reshape(-1, 1) - record, axis=1))

plot_curve(avg_gap)
# plot_diff_curve(average_score_drop_of_methods)
calculate_aoc(avg_gap)
# for method_name, std in aopc_std.items():
#     print('%s: %s' % (method_name, std))
