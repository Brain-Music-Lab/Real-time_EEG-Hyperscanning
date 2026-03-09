import pyxdf
import os
import sys
import time
import math
import mne
import numpy as np
import matplotlib.pyplot as plt

from asrpy import ASR
import scipy.stats as stats
from scipy.fftpack import fft
from scipy import signal
from sklearn.decomposition import PCA
from hypyp import analyses

from mne_lsl.stream import StreamLSL, EpochsStream
from mne_lsl.lsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_streams,
)

from collections import deque

# ----------------- CONFIG -----------------
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4',
            'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Trigger']
ch_types = ['eeg'] * 16 + ['stim']

nChan = len(ch_names) - 1
SR = 500

freq_bands = {
    'Alpha': [7, 14],
}

resting_fif_L = 'raw_L.fif'
resting_fif_F = 'raw_F.fif'
resting_fif = 'raw_rest_combined.fif'

subset_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz']
subset_indices = [ch_names.index(name) for name in subset_names]

# ----------------- BASELINE CONNECTIVITY -----------------
raw_rest_combined = mne.io.read_raw_fif(resting_fif, preload=True)

eventsClean = mne.make_fixed_length_events(raw_rest_combined, start=4, duration=3.0)
epochClean = mne.Epochs(raw_rest_combined, events