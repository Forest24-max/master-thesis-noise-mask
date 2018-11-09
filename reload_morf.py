import numpy as np
import tensorflow as tf
import glob as glob
import os
from experiment_file import get_filenames, prepare_imgs, save_array
import matplotlib.pyplot as plt
import csv

from model_accessor import load_model, top_labels_and_ids, output_node, run_network, output_id_tensor, presoftmax


def load_scores(dir):
    filenames = get_filenames(dir, '*.npy')
    return {os.path.splitext(os.path.basename(filename))[0]: np.load(filename) for filename in filenames}


def plot_curve(curve_of_methods, dpi=192):
    plt.figure(figsize=(1000/dpi, 500/dpi), dpi=dpi)

    baseline = curve_of_methods['random_baseline']
    noise_mask = curve_of_methods['noise_mask']
    del curve_of_methods['random_baseline']
    del curve_of_methods['noise_mask']

    plt.plot(baseline, '--', linewidth=2, label='random')
    for method_name, curve in curve_of_methods.items():
        plt.plot(curve, label=method_name)

    plt.plot(noise_mask, 'k', linewidth=2, label='noise mask')

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='x-large')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    # plt.savefig('media/%s.png' % XX_RF, dpi=dpi, transparent=True)
    plt.show()

def write_csv(curve_of_methods):
    with open('%s.csv' % XX_RF, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        iters = len(next(iter(curve_of_methods.values())))

        writer.writerow(['Method'] + list(range(0, iters)))
        for method_name, curve in curve_of_methods.items():
            writer.writerow([method_name] + list(curve))

def plot_diff_curve(curve_of_methods):
    for method_name, curve in curve_of_methods.items():
        plt.plot(curve[:-1] - curve[1:], label=method_name)

    plt.legend()
    plt.show()

def calculate_aopc(curve_records):
    for method_name, curve in curve_records.items():
        aopc = np.mean(curve)
        print('%s: %s' % (method_name, aopc))

XX_RF = 'morf'
root_dir = 'result/%s-pixel-r-presoftmax-200l-bch100-800' % XX_RF

scores = load_scores(root_dir)

# scores['random_baseline']
avg_drops = {}
aopc_std = {}
for method_name, record in scores.items():
    avg_drops[method_name] = np.mean(record - record[:, 0].reshape(-1, 1), axis=0)
    aopc_std[method_name] = np.std(np.mean(record[:, 0].reshape(-1, 1) - record, axis=1))

write_csv(avg_drops)
# plot_curve(avg_drops)
# plot_diff_curve(average_score_drop_of_methods)
calculate_aopc(avg_drops)
# for method_name, std in aopc_std.items():
#     print('%s: %s' % (method_name, std))
