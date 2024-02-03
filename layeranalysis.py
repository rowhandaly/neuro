import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from sklearn import metrics
from sklearn import linear_model
from open_ephys.analysis import Session
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Load Open Ephys data
directory = '/Users/rowhandaly/Desktop/2024-01-31_23-11-10'
session = Session(directory)
recording = session.recordnodes[0].recordings[0]

# Set the number of signals
x = 16

signals_list = []
for i in range(x):
    signals_list.append([subarray[i] for subarray in recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=10000)])

# Set the parameters for the spectrogram
window_size = 8192  # Set window_size to the length of the shortest signal
n_overlap = window_size // 2  # Ensure n_overlap is less than window_size

# Initialize an empty list to store the power spectral densities
psds = []

# Initialize frequencies to None
frequencies = None

# Assuming signals_list is a list of numpy arrays, where each numpy array is a signal
for i in range(x):
    # Get the signal from signals_list and convert it to a numpy array
    signal = np.array(signals_list[i])

    # Compute the spectrogram
    freq, time, Sxx = spectrogram(signal, fs=500, window='hann', nperseg=window_size, noverlap=n_overlap)

    # If frequencies is None, set it to freq
    if frequencies is None:
        frequencies = freq

    # Calculate the power spectral density
    psd = np.mean(Sxx**2, axis=1)

    # Append the PSD to the list
    psds.append(psd)

# Convert the list of PSDs to a 2D arrayc
psds = np.array(psds)

# Normalize the PSDs to a range of 0 to 1 for each frequency
psds = psds / np.max(psds, axis=0)

# Create a figure and axis
fig, ax = plt.subplots()

# Create a heatmap of the normalized PSDs
cax = ax.imshow(psds, aspect='auto', cmap='viridis', origin='lower')

# Add a colorbar to the figure
fig.colorbar(cax, label='Normalized PSD')

# Set the x-ticks and x-tick labels to be the frequencies
ax.set_xticks(np.arange(0, len(frequencies), 10))
ax.set_xticklabels(frequencies[::10])

# Set the labels for the x-axis and y-axis
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Signal number')

# Set the x-ticks and x-tick labels to be every 20th frequency
ax.set_xticks(np.arange(0, len(frequencies), 1000))
ax.set_xticklabels(frequencies[::1000])

# Set the title of the plot
ax.set_title('Normalized PSDs')

# Display the plot
plt.savefig('/Users/rowhandaly/desktop/RPM.png')

# Set the frequency band limits
bands = [(10, 19), (75, 150)]

def calculate_band_power(a, b, freq, psds):
    # Find the indices of the frequency array where the frequency is between a and b
    freq_indices = np.where((freq >= a) & (freq <= b))[0]  # Take the first element of the tuple

    # Calculate the power for each signal in the specified frequency band
    band_power = psds[:, freq_indices]

    return band_power

# Create a new figure
plt.figure()

# Calculate and plot the power for each signal in each frequency band
for a, b in bands:
    band_power = calculate_band_power(a, b, frequencies, psds)
    avg_band_power = np.mean(band_power, axis=1)  # Calculate the average power for each signal in the band
    plt.plot(avg_band_power, range(1, len(avg_band_power) + 1), label='{}-{} Hz'.format(a, b))  # Plot the average power

plt.xlabel('Power')
plt.ylabel('Signal number')
plt.xlim(0,1)
plt.legend()
plt.savefig('/Users/rowhandaly/desktop/RPF')

# Set the minimum range size
min_range_size = 7

# Initialize the best G and the best range
best_G = -np.inf
best_range = None

# Calculate the power for each signal in each frequency band and calculate the goodness of fit for each possible range of signals
for a, b in bands:
    band_power = calculate_band_power(a, b, freq, psds)
    if band_power.size > 0 and not np.isnan(band_power).any():
        avg_band_power = np.mean(band_power, axis=1)
    else:
        continue
    # Calculate the goodness of fit for each possible range of signals greater than or equal to 7
    for i in range(len(avg_band_power) - min_range_size + 1):
        for j in range(i + min_range_size, len(avg_band_power) + 1):
            # Calculate the linear regression for the range
            model_ab = LinearRegression().fit(np.array(range(i, j)).reshape(-1, 1), avg_band_power[i:j])
            R2_ab = model_ab.score(np.array(range(i, j)).reshape(-1, 1), avg_band_power[i:j])
            s_ab = np.sign(model_ab.coef_[0])
 
            # Initialize R2_g
            R2_g = 0

            # Calculate the linear regression for the rest of the signals
            if i < j and j < len(avg_band_power):
                model_g = LinearRegression().fit(np.array(list(range(i)) + list(range(j, len(avg_band_power)))).reshape(-1, 1), 
                                                 np.concatenate((avg_band_power[:i], avg_band_power[j:])))
                if len(np.array(list(range(i)) + list(range(j, len(avg_band_power)))).reshape(-1, 1)) >= 2:
                    R2_g = model_g.score(np.array(list(range(i)) + list(range(j, len(avg_band_power)))).reshape(-1, 1), 
                                         np.concatenate((avg_band_power[:i], avg_band_power[j:])))
                else:
                    pass
                   # print("Not enough samples for R^2 score")
                s_g = np.sign(model_g.coef_[0])
            else:
                pass
               # print("Invalid range for linear regression")

            # Calculate the goodness of fit
            f = 0.04 * (j - i) + 0.72
            G = s_ab * R2_ab * -s_g**2 * R2_g * f

            if G > best_G:
                best_G = G
                best_range = (i, j)

print('Best range:', best_range)
print('Best G:', best_G)
