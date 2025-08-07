
# python mne-lsl-Trigger.py
import numpy as np

import pyxdf

from mne.io import read_raw_fif
from mne.time_frequency import psd_array_multitaper
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.signal import periodogram, welch

# import mne_lsl
from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL, EpochsStream

from mne_lsl.lsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_streams,
)



import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
# import mne_connectivity

from hypyp import analyses

from asrpy import asr_calibrate, asr_process, clean_windows

import pathlib

import websockets
import asyncio

import time

from pythonosc.udp_client import SimpleUDPClient

# ESP32 IP address and port
esp32_ip_leader = "192.168.0.111"  
esp32_ip_follower = "192.168.0.113"  
esp32_port = 8000

# Create OSC client
osc_client_leader = SimpleUDPClient(esp32_ip_leader, esp32_port)
osc_client_follower = SimpleUDPClient(esp32_ip_follower, esp32_port)
# Define the path to the current script
this_path = pathlib.Path().absolute()

freq_bands = {
    'Delta': [0.5,3.5],
    'Theta': [3.5,7],
    'Alpha':[7,14],
    'Beta': [14, 25]}

nChan = 16
sfreq = 125

from dataclasses import dataclass
@dataclass
class streamClass:
    EEGtimeSeries: np.ndarray
    TRGtimeSeries: np.ndarray
    timeSeries: np.ndarray
    timeStamps: np.ndarray
    SR: float

        
## Load the clean data
root = "C:/Users/thiag/OneDrive - UCB-O365/BML/Hyperdance/TEI"
P_root = "/sub-P001/ses-S001/eeg/"
clean = "sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

cleanstreams, cleansheader= pyxdf.load_xdf(root + P_root + clean)
print('clean stream successfully read')


def detect_falling_edges(markers, timestamps):
    """
    Detect rising edge events in a time series of markers.
    """
    markers = np.asarray(markers).flatten()  # Ensure 1D
    markers = markers / np.max(markers)
    rising_edges = []
    for i in range(1, len(markers)):
        if markers[i-1] < 0.5 and markers[i] > 0.5:
            rising_edges.append(timestamps[i])
    return rising_edges


def getTimeDifference(trg1, trg2, ts1, ts2, SR):
    a = detect_falling_edges(trg1[:,0], ts1)
    b = detect_falling_edges(trg2[:,0], ts2)

# check if the two lists are of the same length
    min_length = min(len(a), len(b))
    a = a[:min_length]
    b = b[:min_length]
#convert a and b to numpy arrays
    a = np.array(a)
    b = np.array(b)

# convert time_differences from seconds to samples
    sample_differece = (a - b) * SR
    return np.round(np.mean(sample_differece))

def fixLatency(streams_lead, streams_follow, SR):
    """
    Fix the latency between two streams by adjusting the timestamps of the second stream.
    
    Parameters:
    streams (list): List containing two streamClass objects.
    SR (float): Sampling rate.
    
    Returns:
    list: List containing the adjusted streams.
    """
    time_difference_samples = int(round(getTimeDifference(streams_lead.TRGtimeSeries, streams_follow.TRGtimeSeries, streams_lead.timeStamps, streams_follow.timeStamps, SR)))

    # Create new stream objects
    New_follow_stream = streamClass(
        EEGtimeSeries=streams_follow.EEGtimeSeries,
        TRGtimeSeries=streams_follow.TRGtimeSeries,
        timeSeries=streams_follow.timeSeries,
        timeStamps=streams_follow.timeStamps,
        SR=streams_follow.SR
    )

    New_lead_stream = streamClass(
        EEGtimeSeries=streams_lead.EEGtimeSeries,
        TRGtimeSeries=streams_lead.TRGtimeSeries,
        timeSeries=streams_lead.timeSeries,
        timeStamps=streams_lead.timeStamps,
        SR=streams_lead.SR
    )

    # Ensure time_difference_samples is within bounds
    mean_diff = time_difference_samples
    if mean_diff > 0:
        print(f"Trimming {mean_diff} samples from the lead stream")
        slice_idx = min(mean_diff, len(streams_follow.timeStamps))
        New_lead_stream.timeSeries = streams_lead.timeSeries[slice_idx:]
        New_lead_stream.EEGtimeSeries = streams_lead.EEGtimeSeries[slice_idx:]
        New_lead_stream.TRGtimeSeries = streams_lead.TRGtimeSeries[slice_idx:]
        New_lead_stream.timeStamps = streams_lead.timeStamps[slice_idx:]
        New_lead_stream.timeStamps = New_lead_stream.timeStamps - slice_idx/New_lead_stream.SR
    elif mean_diff < 0:
        print(f"Trimming {-mean_diff} samples from the follow stream")
        slice_idx = min(-mean_diff, len(streams_follow.timeStamps))
        New_follow_stream.timeSeries = streams_follow.timeSeries[slice_idx:]
        New_follow_stream.EEGtimeSeries = streams_follow.EEGtimeSeries[slice_idx:]
        New_follow_stream.TRGtimeSeries = streams_follow.TRGtimeSeries[slice_idx:]
        New_follow_stream.timeStamps = streams_follow.timeStamps[slice_idx:]
        New_follow_stream.timeStamps = New_follow_stream.timeStamps - slice_idx/New_follow_stream.SR

    # Make both streams have the same length
    min_length = min(len(New_lead_stream.timeStamps), len(New_follow_stream.timeStamps))
    New_lead_stream.timeSeries = New_lead_stream.timeSeries[:min_length]
    New_lead_stream.EEGtimeSeries = New_lead_stream.EEGtimeSeries[:min_length]
    New_lead_stream.TRGtimeSeries = New_lead_stream.TRGtimeSeries[:min_length]
    New_lead_stream.timeStamps = New_lead_stream.timeStamps[:min_length]
    New_follow_stream.timeSeries = New_follow_stream.timeSeries[:min_length]
    New_follow_stream.EEGtimeSeries = New_follow_stream.EEGtimeSeries[:min_length]
    New_follow_stream.TRGtimeSeries = New_follow_stream.TRGtimeSeries[:min_length]
    New_follow_stream.timeStamps = New_follow_stream.timeStamps[:min_length]
    
    return [New_lead_stream, New_follow_stream]


for stream in cleanstreams:
    # Print stream name
    print(f"Stream name: {stream['info']['name'][0]}")
    if stream['info']['name'][0] == 'Cyton_COM11':

        EEG = stream["time_series"][:,0:16]
        TRG = stream["time_series"][:,16:]
        # Normalize TRG values to 0 and 1
        TRG = (TRG > 1).astype(float)
        ALL = stream["time_series"]
        TS = stream["time_stamps"]
        SR = float(stream['info']['effective_srate'])
        stream_follow = streamClass(EEG, TRG, ALL, TS, SR)

    elif stream['info']['name'][0] == 'Cyton_COM10':
        EEG = stream["time_series"][:,0:16]
        TRG = stream["time_series"][:,16:]
        # Normalize TRG values to 0 and 1
        TRG = (TRG > 1).astype(float)
        ALL = stream["time_series"]
        TS = stream["time_stamps"]
        SR = float(stream['info']['effective_srate'])
        stream_lead = streamClass(EEG, TRG, ALL, TS, SR)

SR = (stream_lead.SR + stream_follow.SR)/2

# Fix latency between the two streams
streams_lead, streams_follow = fixLatency(stream_lead, stream_follow, SR)

ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
ch_names_hyp = ['Fp1_1', 'Fp2_1', 'F3_1', 'F4_1', 'T3_1', 'C3_1', 'Cz_1', 'C4_1', 'T4_1', 'T5_1', 'P3_1', 'Pz_1', 'P4_1', 'T6_1', 'O1_1', 'O2_1', 'Fp1_2', 'Fp2_2', 'F3_2', 'F4_2', 'T3_2', 'C3_2', 'Cz_2', 'C4_2', 'T4_2', 'T5_2', 'P3_2', 'Pz_2', 'P4_2', 'T6_2', 'O1_2', 'O2_2']
ch_types_hyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

subset_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz']
subset_indices = [ch_names.index(name) for name in subset_names]

# Create hyperscanning MNE file from clean data
HypClean = np.concatenate((streams_lead.EEGtimeSeries, streams_follow.EEGtimeSeries), axis=1).T

print("Leader EEG shape:", streams_lead.EEGtimeSeries.shape)
print("Follower EEG shape:", streams_follow.EEGtimeSeries.shape)
print("Combined EEG shape:", HypClean.shape)
print("ch_names_hyp:", len(ch_names_hyp))
print("ch_types_hyp:", len(ch_types_hyp))

info = mne.create_info(ch_names=ch_names_hyp, sfreq=SR, ch_types=ch_types_hyp)
rawClean = mne.io.RawArray(HypClean, info)

# get the length of the data in seconds
n_seconds = rawClean.n_times / rawClean.info['sfreq']
print(f"Raw data length: {n_seconds:.2f} seconds")

rawClean.filter(l_freq=0.5, h_freq=45., picks='eeg')
# Calculate baseline Coh from clean EEG data
eventsClean = mne.make_fixed_length_events(rawClean, start=1, duration=4.0)
epochClean = mne.Epochs(rawClean, eventsClean, tmin=0, tmax=4, baseline=None, preload=True)

# Get the data from the epochs
print("Calculating baseline connectivity...")
epochCleanArray = epochClean.get_data(copy=False)

epochBaselineArray_lead = epochCleanArray[:,0:nChan,:]
epochBaselineArray_folllow = epochCleanArray[:,nChan:nChan*2,:]

epochBaselineArray_lead_subset = epochBaselineArray_lead[:, subset_indices, :]
epochBaselineArray_folllow_subset = epochBaselineArray_folllow[:, subset_indices, :]
epochBaselineArray_comb = np.array([epochBaselineArray_lead_subset,epochBaselineArray_folllow_subset])

EpochBaselineComplex = analyses.compute_freq_bands(epochBaselineArray_comb, freq_bands=freq_bands, sampling_rate = sfreq)
ConnCoh = analyses.compute_sync(EpochBaselineComplex, mode = 'coh', epochs_average = False)
BaselineConn= np.mean(ConnCoh[0,:,:,:],axis=0)



# (optional) make sure your asr is only fitted to clean parts of the data
# pre_cleanedL, _ = clean_windows(streams_lead.EEGtimeSeries, streams_lead.SR, max_bad_chans=0.1)
# pre_cleanedF, _ = clean_windows(streams_follow.EEGtimeSeries, streams_follow.SR, max_bad_chans=0.1)

print("Fitting ASR to lead and follow streams...")
# M_l, T_l = asr_calibrate(streams_lead.EEGtimeSeries[:7500,:], streams_lead.SR, cutoff=5)
# M_f, T_f = asr_calibrate(streams_follow.EEGtimeSeries[:7500,:], streams_follow.SR, cutoff=5)

M_l, T_l = 0, 0 # Dummy values for M_l and T_l
M_f, T_f = 0, 0 # Dummy values for M_f and T_f
# ASR calibration is not performed in this example, but you can uncomment the lines above to perform it.

print("ASR calibration done.")

print("Resolving LSL streams...")
bufferSize = 10 # seconds
nsamples = int(SR*bufferSize)

streams = resolve_streams()
print([s.name for s in streams])


# Retrive LSL streams in 1 second buffers
stream_lead = StreamLSL(bufsize=bufferSize, name="Cyton_COM10").connect()
stream_follow = StreamLSL(bufsize=bufferSize, name="Cyton_COM11").connect()

for ch in stream_lead.info['chs']:
    print(ch['ch_name'])


epochs_lead = EpochsStream(
    stream_lead,
    bufsize=20,  # number of epoch held in the buffer
    event_id=256,
    event_channels="P11",
    tmin=-3,
    tmax=3,
    baseline=(None, 0),
    picks="eeg",
).connect(acquisition_delay=0.1)

epochs_follow = EpochsStream(
    stream_follow,
    bufsize=20,  # number of epoch held in the buffer
    event_id=256,
    event_channels="P11",
    tmin=-3,
    tmax=3,
    baseline=(None, 0),
    picks="eeg",
).connect(acquisition_delay=0.1)

while epochs_lead.n_new_epochs < 5:
    time.sleep(5)

np.set_printoptions(threshold=5000)
print("events in P11: ", stream_lead.get_data(picks="P11")[0])
# print(stream_follow.get_data(picks="P11"))
print(f"Lead stream has {epochs_lead.n_new_epochs} epochs available.")
np.set_printoptions(threshold=100)

# Ensure the streams are connected
if not stream_lead.connected or not stream_follow.connected:
    raise RuntimeError("Failed to connect to the LSL streams.")

def main_loop(epochs_lead, epochs_follow, M_l, T_l, M_f, T_f, BaselineConn, nChan, sfreq, freq_bands,
              osc_client_leader, osc_client_follower, bufferSize):

    print("Starting OSC-only main loop...")
    num_repetitions = 0
    elapsed_times = []

    matrixMask = np.zeros([nChan*2, nChan*2])
    nsamples = int(sfreq * bufferSize)
    LeadBuffer = np.zeros([nChan, nsamples])
    FollowBuffer = np.zeros([nChan, nsamples])

    while True:
        start_time = time.time()
        print("Waiting for new epochs...")

        # Wait for new epochs to be available
        lead_epochs = epochs_lead.get_data(n_epochs=5)
        # Now filter using MNE's filter function (if desired)
        # Example: mne.filter.filter_data for numpy arrays
        if lead_epochs is not None and lead_epochs.shape[0] > 0:
            # lead_epochs shape: (n_epochs, n_channels, n_times)
            from mne.filter import filter_data
            for i in range(lead_epochs.shape[0]):
                lead_epochs[i] = filter_data(lead_epochs[i], sfreq=sfreq, l_freq=0.5, h_freq=45.0, verbose=False)


        follow_epochs = epochs_follow.get_data(n_epochs=5)
        # Now filter using MNE's filter function (if desired)
        # Example: mne.filter.filter_data for numpy arrays
        if follow_epochs is not None and follow_epochs.shape[0] > 0:
            # lead_epochs shape: (n_epochs, n_channels, n_times)
            from mne.filter import filter_data
            for i in range(follow_epochs.shape[0]):
                follow_epochs[i] = filter_data(follow_epochs[i], sfreq=sfreq, l_freq=0.5, h_freq=45.0, verbose=False)


        if lead_epochs is None or follow_epochs is None:
            print("No new epochs available, waiting for more data...")
            time.sleep(1)
            continue

        if lead_epochs.shape[0] == 0 or follow_epochs.shape[0] == 0:
            print("No epochs available, waiting for more data...")
            time.sleep(1)
            continue

        print(f"Received {lead_epochs.shape[0]} lead epochs and {follow_epochs.shape[0]} follow epochs")

        # ASR processing (if needed)
        # clean_l = asr_process(lead_epochs, sfreq, M_l, T_l)
        # clean_f = asr_process(follow_epochs, sfreq, M_f, T_f)
        clean_l = lead_epochs
        clean_f = follow_epochs

        print("clean_l.shape:", clean_l.shape)
        print("clean_f.shape:", clean_f.shape)

        clean_l_subset = clean_l[:, subset_indices, :]
        clean_f_subset = clean_f[:, subset_indices, :]
        
        # Concatenate epochs to have the shape (n_epochs, 2*nChan, n_times)
        HypClenEpochs = np.stack([clean_l_subset, clean_f_subset])  # shape: (2, 5, 16, 750)
        print("HypClenEpochs.shape:", HypClenEpochs.shape)

        # Compute connectivity/coherence
        complex_epochs = analyses.compute_freq_bands(HypClenEpochs, freq_bands=freq_bands, sampling_rate=sfreq)
        print("complex_epochs.shape:", complex_epochs.shape)

        conn_coh = analyses.compute_sync(complex_epochs, mode='coh', epochs_average=True)
        print(f"conn_coh: {conn_coh}")
        print(f"BaselineConn: {BaselineConn}")

      
        conn_coh = conn_coh[0, :, :]
        conn_coh = np.where(conn_coh >= BaselineConn, conn_coh, 0)
        # matrixMask[NnChan:2*NnChan, 0:NnChan] = conn_coh[NnChan:2*NnChan, 0:NnChan]

      
        conn_coh[conn_coh == 1] = 0
        print("matrixMask: ", matrixMask)
        print("mean: ",float(np.mean(conn_coh)))
        print("max: ",float(np.max(conn_coh)))
        print("min: ",float(np.min(conn_coh)))

        # Calculate values for OSC
        value = float((np.mean(conn_coh))-0.3)/0.4
        pwm1 = value
        pwm2 = value


        print(f"pwm1: {pwm1}, pwm2: {pwm2}")

        osc_client_leader.send_message("/pwm", [pwm1, pwm2])
        osc_client_follower.send_message("/pwm", [pwm1, pwm2])
        print(f"Sent /pwm {pwm1}, {pwm2}")

        num_repetitions += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        average_time = sum(elapsed_times) / num_repetitions
        print(f"Last execution time: {elapsed_time:.4f} seconds")
        print(f"Average execution time over {num_repetitions} repetitions: {average_time:.4f} seconds")

        # Wait for 500ms before sending the next update
        time.sleep(0.5)

# --- At the end of your script, call the main loop:
if __name__ == "__main__":
    main_loop(
        epochs_lead, epochs_follow, M_l, T_l, M_f, T_f, BaselineConn,
        nChan, sfreq, freq_bands, osc_client_leader, osc_client_follower, bufferSize
    )

