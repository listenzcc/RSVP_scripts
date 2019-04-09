# code: utf-8

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import time

##############
# Parameters #
##############
print('Initing parameters.')
# Results pdf path
result_dir = os.path.join('D:\\', 'RSVP_MEG_experiment', 'scripts',
                          'MotionImaging', 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

pdf_path = os.path.join(result_dir, 'foo_%s.pdf' %
                        time.strftime('%Y-%m-%d-%H-%M-%S'))
npz_path = os.path.join(result_dir, 'npz_%s.npz')


figures = []

for freq in [180, 240]:
    # load variables from npz file
    data = np.load(npz_path % str(freq))
    scores = data['scores'].tolist()
    w_times = data['w_times']

    f = plt.figure()

    for clf_name in scores.keys():
        plt.plot(w_times, np.mean(np.mean(scores[clf_name], 0), 0),
                 label=clf_name)

    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time %sHz' % str(freq))
    plt.legend(loc='lower right')
    figures.append(f)

    # Saving into pdf
    print('Saving into pdf.')
    with PdfPages(pdf_path) as pp:
        for f in figures:
            pp.savefig(f)

# Finally, we show all figures.
plt.show()
