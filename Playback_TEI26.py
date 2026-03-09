import pyxdf

import os
import sys
import time
import json
import asyncio
import threading

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

from pythonosc.udp_client import SimpleUDPClient

import websockets


WS_HOST = "0.0.0.0"
WS_PORT = 8080
_ws_clients = set()
_ws_loop = None


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    client_addr = getattr(websocket, "remote_address", "unknown")
    print(f"Vizaj WebSocket client connected: {client_addr}")
    try:
        await websocket.wait_closed()
    finally:
        _ws_clients.discard(websocket)
        print(f"Vizaj WebSocket client disconnected: {client_addr}")


async def _broadcast_text(message_text):
    if not _ws_clients:
        return
    disconnected = []
    for client in list(_ws_clients):
        try:
            await client.send(message_text)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        _ws_clients.discard(client)


def send_to_vizaj(payload):
    if _ws_loop is None:
        return
    message_text = json.dumps(payload)
    asyncio.run_coroutine_threadsafe(_broadcast_text(message_text), _ws_loop)


def _start_websocket_server():
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    server = websockets.serve(_ws_handler, WS_HOST, WS_PORT)
    _ws_loop.run_until_complete(server)
    print(f"Vizaj WebSocket server started on ws://{WS_HOST}:{WS_PORT}")
    _ws_loop.run_forever()

# ESP32 IP address and port
esp32_ip_leader = "192.168.0.111"  
esp32_ip_follower = "192.168.0.113"  
esp32_port = 8000

# Create OSC client
try:
    osc_client_leader = SimpleUDPClient(esp32_ip_leader, esp32_port)
    print(f"Connected to leader at {esp32_ip_leader}:{esp32_port}")
except Exception as e:
    print(f"Warning: Could not connect to OSC clients: {e}")


try:
    osc_client_follower = SimpleUDPClient(esp32_ip_follower, esp32_port)
    print(f"Connected to follower at {esp32_ip_follower}:{esp32_port}")
except Exception as e:
    print(f"Warning: Could not connect to OSC clients: {e}")


# create a mne object for each stream and filter it
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Trigger']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'stim']
subset_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'Cz', 'P3', 'P4', 'Pz']

nChan = len(ch_names) - 1

SR = 500

freq_bands = {
    # 'Delta': [0.5,3.5],
    # 'Theta': [3.5,7],
    'Alpha':[7,14],
    # 'Beta': [14, 25]}
}

resting_fif_L = 'raw_L.fif'
resting_fif_F = 'raw_F.fif'
resting_fif = 'raw_rest_combined.fif'

# read in the .fif file
raw_rest_combined = mne.io.read_raw_fif(resting_fif, preload=True)

eventsClean = mne.make_fixed_length_events(raw_rest_combined, start=4, duration=3.0)
epochClean = mne.Epochs(raw_rest_combined, eventsClean, tmin=-3, tmax=3, baseline=None, preload=True)

## Get baseline connectivity
print("Calculating baseline connectivity...")
epochCleanArray = epochClean.get_data(copy=False)
print(nChan)
epochBaselineArray_lead = epochCleanArray[:,0:nChan,:]
epochBaselineArray_folllow = epochCleanArray[:,nChan:nChan*2,:]

print(epochBaselineArray_lead.shape)
print(epochBaselineArray_folllow.shape)
# epochBaselineArray_lead_subset = epochBaselineArray_lead[:, subset_indices, :]
# epochBaselineArray_folllow_subset = epochBaselineArray_folllow[:, subset_indices, :]
epochBaselineArray_comb = np.array([epochBaselineArray_lead,epochBaselineArray_folllow])

EpochBaselineComplex = analyses.compute_freq_bands(epochBaselineArray_comb, freq_bands=freq_bands, sampling_rate = SR)
ConnCoh = analyses.compute_sync(EpochBaselineComplex, mode = 'coh', epochs_average = False)
BaselineConn= np.mean(ConnCoh[0,:,:,:],axis=0)

## get ASR template from resting data
# raw_resting_L = mne.io.read_raw_fif(resting_fif_L, preload=True)
# raw_resting_F = mne.io.read_raw_fif(resting_fif_F, preload=True)

raw_resting_L = mne.io.read_raw_fif('raw_L.fif', preload=True)
raw_resting_F = mne.io.read_raw_fif('raw_F.fif', preload=True)

# Select only EEG channels for ASR fitting
raw_resting_L.pick_types(eeg=True)
raw_resting_F.pick_types(eeg=True)

# Print data info for debugging
print('.............................')
print(f"Lead data duration: {raw_resting_L.times[-1]:.2f} seconds")
print(f"Follow data duration: {raw_resting_F.times[-1]:.2f} seconds")
print(f"EEG channels: {raw_resting_L.ch_names}")
print(f"EEG subset: {subset_names}")
print(f"Lead data shape: {raw_resting_L.get_data().shape}")
print(f"Follow data shape: {raw_resting_F.get_data().shape}")
print('.............................')
# Use blocksize=5 (5 seconds) to handle shorter recordings
asr_lead = ASR(sfreq=raw_resting_L.info["sfreq"], cutoff=10)
asr_lead.fit(raw_resting_L)


asr_follow = ASR(sfreq=raw_resting_F.info["sfreq"], cutoff=10)
asr_follow.fit(raw_resting_F)



#####


print("Resolving LSL streams...")
bufferSize = 10 # seconds
nsamples = int(SR*bufferSize)

streams = resolve_streams()
print([s.name for s in streams])


# Retrive LSL streams in 1 second buffers
combined_stream = StreamLSL(bufsize=bufferSize, name="Simple2_Combined").connect()


# for ch in combined_stream.info['chs']:
#     print(ch['ch_name'])


epochs_raw = EpochsStream(
    combined_stream,
    bufsize=2,  # number of epoch held in the buffer
    event_id=1,
    event_channels="Trigger",
    tmin=-1.5,
    tmax=1.5,
    baseline=(None, 0),
    picks="eeg",
).connect(acquisition_delay=0.1)



while epochs_raw.n_new_epochs < 2:
    time.sleep(1)

np.set_printoptions(threshold=5000)
print("events in Trigger: ", combined_stream.get_data(picks="Trigger")[0])
# print(stream_follow.get_data(picks="P11"))
print(f"Combined stream has {epochs_raw.n_new_epochs} epochs available.")
np.set_printoptions(threshold=100)

# Ensure the streams are connected
if not combined_stream.connected:
    raise RuntimeError("Failed to connect to the LSL streams.")


subset_indices_L = [ch_names.index(name) for name in subset_names]
subset_indices_F = [ch_names.index(name) + nChan for name in subset_names]

plt.ion()
history_length = 400
coh_history = deque(maxlen=history_length)

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [], lw=2)
ax.axhline(0, color='k', linestyle='--', linewidth=1)

ax.set_ylim(-0.05,1.5)
ax.set_xlim(0, history_length)
ax.set_xlabel("Epochs")
ax.set_ylabel("Scaled Inter-brain Coupling")
ax.set_title("Real-time Inter-Brain Coupling (Spectral Coherence - Alpha Band)")
ax.set_xticks([])
ax.axhline(0.8, color='r', linestyle='--', linewidth=1)

plt.show()


# Now you can process the data in real-time as it arrives
print("Processing incoming data...")
while True:
    if epochs_raw.n_new_epochs > 0:
        n_new = epochs_raw.n_new_epochs
        # print(f"New epochs available: {n_new}")
        # new_epochs_data = epochs_raw.get_data()
        try:
            print(".......................................")
            new_epochs_data = epochs_raw.get_data()
            # print(f"Received new epochs with shape: {new_epochs_data.shape}")
            
            # separate lead and follow data
            nChan = 16  # 16 EEG channels per person
            # epochs_raw returns (n_epochs, n_channels, n_times)
            if len(new_epochs_data.shape) != 3:
                print(f"Unexpected data shape: {new_epochs_data.shape}")
                continue
            
                            # Already 3D: (n_epochs, n_channels, n_times)
            new_epochs_lead = new_epochs_data[:, 0:nChan, :]
            new_epochs_follow = new_epochs_data[:, nChan:nChan*2, :]


            # print(f"Lead shape: {new_epochs_lead.shape}, Follow shape: {new_epochs_follow.shape}")

            # Concatenate epochs to have the shape (2, n_epochs, nChan, n_times)
            HypClenEpochs = np.array([new_epochs_lead, new_epochs_follow])
            # print("HypClenEpochs.shape:", HypClenEpochs.shape)

            # Compute connectivity/coherence
            complex_epochs = analyses.compute_freq_bands(HypClenEpochs, freq_bands=freq_bands, sampling_rate=SR)

            # print("Complex epochs shape (freq bands, n_epochs, n_channels, n_channels):", complex_epochs.shape)
            conn_coh = analyses.compute_sync(complex_epochs, mode='coh', epochs_average=False)
            # print("Connectivity shape (freq bands, n_epochs, n_channels, n_channels):", conn_coh.shape)

            BaselineConn_subset = BaselineConn[subset_indices_L][:, subset_indices_F]  # shape: (subset_channels, subset_channels)
            # print("BaselineConn shape (n_channels, n_channels): ", BaselineConn.shape)
            # print("BaselineConn_subset shape (subset_channels, subset_channels): ", BaselineConn_subset.shape)

            avg_epochs_conn_coh = np.mean(conn_coh, axis=1)  # average across epochs, shape: (freq bands, n_channels, n_channels)
            # print("Average connectivity across epochs shape (freq bands, n_channels, n_channels): ", avg_epochs_conn_coh.shape)
            conn_coh_alpha = avg_epochs_conn_coh[0,:,:]  # shape: (freq bands, subset_channels, subset_channels)
            # print("conn_coh_alpha shape (subset_channels, subset_channels): ", conn_coh_alpha.shape)
            conn_coh_subset = conn_coh_alpha[subset_indices_L][:, subset_indices_F]  # shape: (subset_channels, subset_channels)
            # print("conn_coh_subset shape (subset_channels, subset_channels): ", conn_coh_subset.shape)
            conn_coh_baseline_corr = np.where(conn_coh_subset >= BaselineConn_subset, conn_coh_subset, 0)

            mean_conn_coh = np.mean(conn_coh_baseline_corr)  # average across all connections in the subset
  
            # print("mean: ",mean_conn_coh)
            print("IBC: ",float(mean_conn_coh))
    
            # Calculate values for OSC
            value = (mean_conn_coh*10)  # Normalize to 0-1 range based on expected min/max
            pwmValue = np.clip(value, 0.1, 1.0)  # Ensure the value is within the desired range
            pwm = pwmValue

            print(f"pwm: {value}")

            osc_client_leader.send_message("/pwm", [pwm, pwm])
            osc_client_follower.send_message("/pwm", [pwm, pwm])
            # print(f"Sent /pwm {pwm}, {pwm}")
            
            # print("Averaged connectivity (freq bands, n_channels, n_channels):", conn_coh)
            # print(f"conn_coh: {conn_coh}")
            # print(f"BaselineConn: {BaselineConn}")

            # subtract baseline connectivity

            # print("Baseline-subtracted connectivity (freq bands, n_channels, n_channels):", conn_coh_baseline_subtracted)
            # get conn_coh for subset electrodes

            # print(f"Subset electrode indices (Lead): {subset_indices_L}")
            # print(f"Subset electrode indices (Follow): {subset_indices_F}")

            
            
            # print("Baseline-subtracted connectivity for subset electrodes (freq bands, subset_channels, subset_channels):", conn_coh_baseline_subset)



            # n_subset = len(subset_indices)
            # interpersonal_mask = np.ones((n_subset, n_subset), dtype=bool)
            # np.fill_diagonal(interpersonal_mask, 0)  # zero out the diagonal to exclude intra-personal connections
            # avg_interpersonal_coh = np.mean(conn_coh_subset_baseline_subtracted[0][interpersonal_mask])
            # print(f"Average interpersonal coherence (baseline-subtracted): {avg_interpersonal_coh}")


            # coh_history.append(mean_conn_coh)  # Assuming we're interested in the first frequency band (e.g., Alpha)
            coh_history.append(value)
            line.set_xdata(range(len(coh_history)))
            line.set_ydata(list(coh_history))
            ax.set_xlim(0, history_length)
            plt.pause(0.01)



        except Exception as e:
            print(f"Error processing epochs: {e}")
            time.sleep(0.5)
            continue



    else:
        time.sleep(0.5)  # Sleep briefly to avoid busy-waiting
