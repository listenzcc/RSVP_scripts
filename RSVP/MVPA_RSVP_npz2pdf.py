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
                          'RSVP', 'results')
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

for pre_ in ['RSVP_MEG_eachTrain_OneClassSVM_2019-05-07-09-00-53']:
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
        for score_method in ['Auc', 'Rec', 'Acc']:
            tmp_scores = data.get('scores_%s' % score_method, 'null')
            if tmp_scores == 'null':
                continue
            scores = tmp_scores.tolist()
            w_times = data['w_times']
            print(w_times)

            f = plt.figure()

            if type(scores) == 'dict':
                for clf_name in scores.keys():
                    plt.plot(w_times, np.mean(np.mean(scores[clf_name], 0), 0),
                             label=clf_name)
            else:
                plt.plot(w_times, np.mean(np.mean(scores, 0), 0),
                         label='OneClassSVM')

            plt.axvline(0, linestyle='--', color='k', label='Onset')
            # plt.axhline(0.5, linestyle='-', color='k', label='Chance')
            plt.xlabel('time (s)')
            plt.ylabel('score of %s' % score_method)
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
