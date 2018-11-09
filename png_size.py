from glob import glob
import numpy as np
import os

root_dir = 'result/methods_final_2_large'
scan_dirs = glob(root_dir + '/*')

print(scan_dirs)

filesizes = []
for sub_dir in scan_dirs:
    filenames = glob(sub_dir + '/*-gray.png')
    for filename in filenames:
        filesizes.append(os.path.getsize(filename))

    size_mean = np.mean(filesizes)
    size_std = np.std(filesizes)

    print('%s: mean : %skb std: %skb' % (sub_dir, size_mean / 1024, size_std / 1024))

    filesizes.clear()