print("running")

# Print the path of the Python interpreter being used
import sys
print("Python executable being used:", sys.executable)

import matplotlib.pyplot as plt
import numpy as np
from open_ephys.analysis import Session
import os
import matplotlib.ticker as ticker
from scipy.io import loadmat
import syncopy as spy
from syncopy.datatype.continuous_data import AnalogData
import json
cfg = spy.StructDict()
import psutil
import time

# Function to print CPU and memory usage
def print_resource_usage():
    print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    memory_info = psutil.virtual_memory()
    print(f"Memory usage: {memory_info.percent}%")
    # print(f"Available memory: {memory_info.available / (1024 ** 2)} MB")
    # print(f"Used memory: {memory_info.used / (1024 ** 2)} MB")

print("packages loaded")

"""
LOAD PARAMETERS AND DATA FROM JSON CONFIG FILE:
general parameters: num_signals, trials (1 sec @ 30000 hz), dead channels, trial/channel number of single psd
matlab parameters: raw data array
open ephys binary parameters: load TTL text file, load timestamps file, enter sync, load directory
"""

# load json config file
with open('/home/rdaly12/Desktop/configH91421.json') as f:
    config = json.load(f)

# general parameters
data_source = config['data_source']
num_signals = config['num_signals']
trials = config['trials']
dead_channels = config['dead_channels']
tnum = config['tnum']
cnum = config['cnum']
alphabeta_lower = config['alphabeta_lower']
alphabeta_upper = config['alphabeta_upper']
gamma_lower = config['gamma_lower']
gamma_upper = config['gamma_upper']

if data_source == 'matlab':
    # matlab parameters
    file = config['file']
elif data_source == 'open_ephys_binary':
    # open ephys parameters
    ttl = config['ttl']
    timestamps = config['timestamps']
    sync = config['sync']
    directory = config['directory']

print("parameters loaded")


"""
DATA FORMATTING:
format of signals_list: signals_list[i] is the ith trial, contains 30000 sublists, each with num_signals signals
"""
# matlab
def matlab(file):
    signals_list = []
    mat_data = loadmat(file)
    values_list = list(mat_data.values())
    signals = values_list[3]
    for i in range(trials):
        signals_list.append([])
        signals_list[i].append(signals[30000*i:30000*(i+1)])
    return signals_list

# open ephys binary
def open_ephys_binary(ttl, timestamps, sync, directory):
    print_resource_usage()
    # load messages
    # array is file containing TTL messages
    array = np.load(ttl)
    str_array = [item.decode() for item in array]
    split_array = [item.split() for item in str_array]
    messages = split_array
    
    # load timestamps
    # timestamps is file containing sample numbers, same length as array
    timestamps = np.load (timestamps)
    
    # enter sync
    sync = sync

    # set print settings
    np.set_printoptions(threshold=np.inf)

    # find beginning of trials
    w = 0
    for i in range(len(messages)):
        if 'IsAcquiring' in messages[i]:
            w = i
   
    # combine messages and timestamps
    mandt = []
    for i in range(w, len(messages)):
        if len(messages)-1 != i:
            mandt.append([])
            mandt[i-w].append(i)
            mandt[i-w].append(messages[i])
            mandt[i-w].append(timestamps[i])
            mandt[i-w].append(int(timestamps[i+1])-int(timestamps[i]))
   
    # find good trials
    ms = []
    for i in range(len(mandt)):
        if 'GoodOrBadTrial' in mandt[i][1][0] and '1' in mandt[i][1][1]:
            ms.append(mandt[(i-2)])

    # get samples
    samples = []
    for i in range(len(ms)):
        samples.append(int(ms[i][2]-(15000+sync)))


    # load open ephys session
    directory = directory
    session = Session(directory)
    recording = session.recordnodes[0].recordings[0]
    
    print("session loaded")
    
    return recording, samples
    
#     # get the first x signals from the recording
#     slist = []
#     signals_list = []
#     for g in range(trials):
#         # REVERSE SIGNALS!
#         print("getting signal trial", g)
#         slist.append([])
#         for i in range(num_signals):
#             slist[g].append([subarray[-i] for subarray in recording.continuous[0].get_samples(start_sample_index = samples[g], end_sample_index = samples[g]+30000)])
   
#     print("loop ended")
    
#     for g in range(trials):
#         print("adding signal trial", g)
#         signals_list.append([])
#         for w in range(30000): 
#             signals_list[g].append([])
#             for i in range(num_signals):
#                 signals_list[g][w].append(slist[g][i][w])
   
#     return signals_list

print("functions defined")

if data_source == 'matlab':
    print("matlab data source")
    signals_list = matlab(file)
elif data_source == 'open_ephys_binary':
    print("binary data source")
    recording, samples = open_ephys_binary(ttl, timestamps, sync, directory)

# # delete dead channels
# for t in dead_channels:
#     for i in range(trials):
#         for w in range(30000):
#             del signals_list[i][w][t]
# # might require debugging if # of dead channels > 1
# del_chs = len(dead_channels) # number of deleted channels 250
# num_signals = num_signals - del_chs


"""
COMPUTATION AND GRAPHING:
1. produces PSD for specified trials/channels
2. produces PSD averaged across all trials for each channel
3. produces RPM for all trials/channels
4. produces RPF for all trials/channels
"""
# get the first x signals from the recording
slist = []
signals_list = []
   

# initialize list to store all psds
plot_data1 = []

for w in range(trials):
    print (f"trial {w+1}")
    print_resource_usage()
    
    # REVERSE SIGNALS!
    slist.append([])
    for i in range(num_signals):
        slist[w].append([subarray[-i] for subarray in recording.continuous[0].get_samples(start_sample_index = samples[w], end_sample_index = samples[w]+30000)])

    signals_list.append([])
    for g in range(30000): 
        signals_list[w].append([])
        for i in range(num_signals):
            signals_list[w][g].append(slist[w][i][g])
    
    # load trial and format as 2d numpy array
    trial_signals_list = signals_list[0]


    signals = np.array(trial_signals_list)
    signals = np.squeeze(signals)

    # initialize list to store psds per trial
    psds = []

    # define frequency range
    freq = np.linspace(0, 150, 150)  # frequencies from 0 to 150 Hz with a step of 1 Hz
    freq = freq.flatten()

    # pass data to analogdata constructor
    analog_data = AnalogData(data=signals, samplerate=30000)

    # pass analog data to freqanalysis method
    fft_spectra = spy.freqanalysis(analog_data, output='pow', method='mtmfft', foi=np.arange(0, 150), tapsmofrq=2, pad = 'nextpow2')
    psd = fft_spectra.data

    # append psd values to psds
    for i in range(num_signals):
        psd_single = psd[0, 0, :, i].flatten()
        psds.append(psd_single)  # Append the PSD values to psds

    plt.figure()
    
    # plot single trials
    if w == tnum:
        for i in range(len(tnum)):
            ts1 = psd[0,0,:,cnum]
            plt.plot(freq, ts1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Power Spectral Density')
        plt.savefig(f'/home/rdaly12/Desktop/Channel{cnum[0]+1}Trial{tnum[0]+1}.png')
        # plt.savefig(f'/Users/rowhandaly/desktop/Channel{cnum[0]+1}Trial{tnum[0]+1}.png')
        plt.close()

    # convert list of psds to 2d array
    psds = np.array(psds)
    plot_data1.append(psds)
    
    
"""
COMBINE TRIALS
"""

# calculate averages across trials
average_data1 = np.mean(plot_data1, axis=0)


for i in range(num_signals):
    plt.figure()
    plt.plot(freq, average_data1[i])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectral Density')
    plt.savefig(f'/home/rdaly12/Desktop/AVGpsdChannel{i+1}.png')
    # plt.savefig(f'/Users/rowhandaly/desktop/AvgPSDs/Channel{i+1}.png')
    plt.close()

plt.figure()

average_data1 = average_data1 / np.max(average_data1, axis=0)

fig, ax = plt.subplots()
cax = ax.imshow(average_data1, aspect='auto', cmap='viridis', origin='lower')
fig.colorbar(cax, label='Normalized PSD')

# set the x-ticks and x-tick labels to be the frequencies and format them
ax.set_xticks(np.arange(0, len(freq), 50))
ax.set_xticklabels(freq[::50])
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# set the y-ticks and y-tick labels to be the signal numbers and format them
ax.set_yticks([num_signals, 0])
ax.set_yticklabels([num_signals, 1])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# set the labels for the x-axis and y-axis
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Signal number')

# set the title of the plot
ax.set_title('Normalized PSDs')

# display the plot
plt.savefig('/home/rdaly12/Desktop/RPM-average.png')

# plt.savefig('/Users/rowhandaly/desktop/RPM-average.png')

# create a new figure
plt.figure()

# relative power function
rpfab = []
rpfg = []
for i in range(num_signals):
    # Calculate the average power in the alpha/beta and gamma bands
    alphabeta_power = np.mean(average_data1[i, (freq >= alphabeta_lower) & (freq <= alphabeta_upper)])
    rpfab.append(alphabeta_power)
    gamma_power = np.mean(average_data1[i, (freq >= gamma_lower) & (freq <= gamma_upper)])
    rpfg.append(gamma_power)

# make num_signals an array
signal_numbers = np.arange(num_signals)

plt.figure()
plt.plot(rpfab, signal_numbers, label = 'Alpha-Beta Band')
plt.plot(rpfg, signal_numbers, label = 'Gamma Band')
plt.xlabel('Relative Power')
plt.ylabel('Signal number')
plt.title('Relative Power in Alpha/Beta and Gamma Bands')
plt.savefig('/home/rdaly12/Desktop/RPF-average.png')
# plt.savefig('/Users/rowhandaly/desktop/RPF-average.png')


# create a new figure
plt.figure()

# Filter out points below 0.1 relative power
filtered_rpfab = [power for power in rpfab if power >= 0.1]
filtered_rpfg = [power for power in rpfg if power >= 0.1]
filtered_signal_numbers_ab = [i for i, power in enumerate(rpfab) if power >= 0.1]
filtered_signal_numbers_g = [i for i, power in enumerate(rpfg) if power >= 0.1]

plt.figure()
plt.plot(filtered_rpfab, filtered_signal_numbers_ab, label='Alpha-Beta Band')
plt.plot(filtered_rpfg, filtered_signal_numbers_g, label='Gamma Band')
plt.xlabel('Relative Power')
plt.ylabel('Signal number')
plt.title('Relative Power in Alpha/Beta and Gamma Bands')
plt.legend()
plt.savefig('/home/rdaly12/Desktop/filtered-RPF-average.png')
# plt.savefig('/Users/rowhandaly/desktop/filtered - RPF-average.png')


# end of code
print("code complete")
# os.system('afplay /System/Library/Sounds/Ping.aiff'
