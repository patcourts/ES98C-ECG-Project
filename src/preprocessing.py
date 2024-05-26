from scipy.signal import butter, filtfilt #for the butterworth filter
import pickle
import gzip
import numpy as np
from tqdm import tqdm


# Load allowed_patients from the compressed pickle file
try:
    with gzip.open('allowed_patients.pkl.gz', 'rb') as f:
        allowed_patients = pickle.load(f)
    print(len(allowed_patients))  # Now you can use allowed_patients in your script
except FileNotFoundError:
    print("The file allowed_patients.pkl.gz was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


#
no_patients = allowed_patients.count_patients()
sample_length = allowed_patients.get_patients(0).get_desired_sample_length()


signals = np.zeros(shape=(no_patients, sample_length))
for i in tqdm(range(0, no_patients)):
    signal = allowed_patients.get_patients(i).read_signal(['v1']).reshape(-1)
    signals[i] = signal


#define the cutoff frequency for the Butterworth filter
cutoff_freq = 1  # Adjust this value based on the characteristics of your signal

# Normalize the cutoff frequency
nyquist_freq = 500  # Nyquist frequency is half the sampling frequency
normalized_cutoff = cutoff_freq / nyquist_freq

#order
order = 3

def butterworth_detrending(signal, cut_off_freq, nyquist_freq, order):
    normalised_cut_off = cut_off_freq/nyquist_freq
    
    b, a = butter(order, normalized_cutoff, btype='low')

    baseline = filtfilt(b, a, signal)#thought this would give the detrended signal but gives a very good fit here
    detrended_signal = signal-baseline
    return detrended_signal


butterworth_detrended_signals = np.zeros(shape=(no_patients, sample_length))
trends = np.zeros(shape=(no_patients, sample_length))

for i, signal in enumerate(signals):
    butterworth_detrended_signals[i] = butterworth_detrending(signal, cutoff_freq, nyquist_freq, order)

#only does it for one singular channel