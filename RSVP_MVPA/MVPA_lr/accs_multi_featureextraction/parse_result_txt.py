#! code: utf-8

import os
import time
import matplotlib.pyplot as plt

print(time.ctime())

working_dir = os.path.join('d:\\', 'RSVP_MEG_experiment', 'scripts',
                           'RSVP_MVPA', 'MVPA_lr',
                           'accs_multi_featureextraction')


def pntlines(lines):
    for j, e in enumerate(lines):
        print(j, ':', e, end='')


def pntline(lines, i, end=''):
    print('%d:' % i, lines[i], end=end)


print(working_dir)
os.listdir(working_dir)


def parse_txtfile(txt_filename, axe, working_dir=working_dir):
    with open(os.path.join(working_dir, txt_filename), 'r') as f:
        lines = f.readlines()

    recalls = []
    preciss = []
    accs = []

    for j in range(1, len(lines), 20):
        pntline(lines, j)
        pntline(lines, j+3)
        pntline(lines, j+6)
        print()

        recall = float(lines[j+3].split()[2])
        recalls.append(recall)

        precis = float(lines[j+3].split()[1])
        preciss.append(precis)

        acc = float(lines[j+6].split()[1])
        accs.append(acc)

    axe.plot(recalls, label='recall')
    axe.plot(preciss, label='precis')
    axe.plot(accs, label='acc')
    axe.legend()
    axe.set_title(txt_filename)

    return recalls, preciss


fig, axes = plt.subplots(2, 2)

recalls, preciss = parse_txtfile('accs_meg_S01.txt', axes[0][0])
recalls, preciss = parse_txtfile('accs_meg_S02.txt', axes[0][1])
recalls, preciss = parse_txtfile('accs_eeg_S01.txt', axes[1][0])
recalls, preciss = parse_txtfile('accs_eeg_S02.txt', axes[1][1])

fig.savefig(os.path.join(working_dir, 'accs_fig.tif'))

plt.show()
