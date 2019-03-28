# coding: utf-8

import functools
import mne
import os
import pickle
import time


def time_it(fn):

    @functools.wraps(fn)
    def new_fn(*args, **kws):
        print('-' * 60)
        start = time.time()
        result = fn(*args, **kws)
        end = time.time()
        duration = end - start
        print('%s seconds are consumed in executing function:\n\t%s%r' %
              (duration, fn.__name__, args))
        return result
    return new_fn


@time_it
def get_epochs(fname, event_id, tmin, t0, tmax,
               freq_l=1, freq_h=10, decim=1):
    # Make defaults
    baseline = (tmin, t0)
    reject = dict(mag=5e-12, grad=4000e-13)

    # Prepare rawobject
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.filter(freq_l, freq_h, fir_design='firwin')
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           eog=False, stim=False, exclude='bads')

    # Get events
    events = mne.find_events(raw)

    # Get epochs
    epochs = mne.Epochs(raw, event_id=event_id, events=events,
                        decim=decim, tmin=tmin, tmax=tmax,
                        picks=picks, baseline=baseline,
                        reject=reject, preload=True)
    return epochs


@time_it
def save_file(obj, path):
    # pickle can not use 'with open(..) as f'
    # do not know why

    # make sure path is legal
    def legal_path(path):
        if not path.endswith('.pkl'):
            path += '.pkl'
        if not os.path.exists(os.path.dirname(path)):
            print('%s does not exist, mkdir.' %
                  os.path.dirname(path))
            os.mkdir(os.path.dirname(path))
        return path

    path = legal_path(path)
    print('Touch %s' % path)
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()
    print('%s pickled.' % path)


@time_it
def load_file(path):
    print('Loading %s' % path)
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    print('%s loaded.' % path)
    return obj
