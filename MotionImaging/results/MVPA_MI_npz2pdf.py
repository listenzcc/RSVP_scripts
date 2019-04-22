# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import time

'''
This script is to draw scores into pictures,
and then draw the pictures into pdf file.

It can automatically detect score npz files dirs startswith pre_,
and make one pdf file for each dir, name pdf file as dir pre.

Paremeters:
    result_dir: the folder path where npz file dirs are.
    pre_: the prefix of which dirs startswith.
'''
##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
result_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                          'MotionImaging', 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def listdir(dirpath, pre=''):
    # call for os.listdir and shrink the results as:
    # 1. isidr
    # 2. startswith pre
    subdirs = [e for e in os.listdir(dirpath) if os.path.isdir(
        os.path.join(dirpath, e)) and e.startswith(pre)]
    return subdirs


npz_path = os.path.join(result_dir, 'npz_%s.npz')

for pre_ in ['MI_MEG_middleTrain',
             'MI_EEG_middleTrain']:
    print(pre_)
    pdf_path = os.path.join(result_dir,
                            '%s_%s.pdf' % (
                                pre_, time.strftime('%Y-%m-%d-%H-%M-%S')))
    target_dir = listdir(result_dir, pre=pre_)
    assert(len(target_dir) == 1)
    target_dir = os.path.join(result_dir, target_dir[0])

    figures = []

    npz_fname_list = os.listdir(target_dir)
    npz_fname_list.sort(key=lambda s: s[4:-4])
    [print(e) for e in npz_fname_list]

    for npz_fname in npz_fname_list:
        # load variables from npz file
        data = np.load(os.path.join(target_dir, npz_fname))
        scores = data['scores'].tolist()
        w_times = data['w_times']

        f = plt.figure()

        for clf_name in scores.keys():
            s = scores[clf_name]
            print(s.shape)
            while len(s.shape) > 1:
                s = np.mean(s, 0)
            plt.plot(w_times, s, label=clf_name)

        plt.axvline(0, linestyle='--', color='k', label='Onset')
        plt.axhline(0.5, linestyle='-', color='k', label='Chance')
        plt.xlabel('time (s)')
        plt.ylabel('classification accuracy')
        plt.title('Classification score over time %sHz' % npz_fname[4:-4])
        plt.legend(loc='lower right')
        figures.append(f)

        # Saving into pdf
        print('Saving into pdf.')
        with PdfPages(pdf_path) as pp:
            for f in figures:
                pp.savefig(f)

# Finally, we show all figures.
# plt.show()
plt.close('all')
