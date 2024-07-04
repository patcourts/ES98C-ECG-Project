import numpy as np
from scipy.signal import butter, filtfilt #for the butterworth filter
import pywt #for DWT


#high-pass butterworth filter removes baseline wander
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

#band-pass butterworth filter to remove noise
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def denoise(signal, butter=False, DWT=False, normalise=False):
    
    sample_freq = 1000
    
    if butter and DWT:
        print('only one filter can be applied to the data')
        return None
        
    if butter:
        cutoff_frequency = 0.5  # Cutoff frequency in Hz (remove frequencies below this)
        b, a = butter_highpass(cutoff_frequency, sample_freq)
        baseline_removed_signal = filtfilt(b, a, signal)
        
        lowcut = 1 # Lower bound of the band-pass filter
        highcut = 50  # Upper bound of the band-pass filter
        b, a = butter_bandpass(lowcut, highcut, sample_freq)
        denoised_signal = filtfilt(b, a, baseline_removed_signal)
        
    elif DWT:
        coeffs = pywt.wavedec(signal, wavelet='db4')
        set_to_zero = [0, 1, 2, 3, 4]
        level_to_zero = 9
        for i in range(0, len(coeffs)):
            if i in set_to_zero or i > level_to_zero:
                coeffs[i] = np.zeros_like(coeffs[i])
        denoised_signal = pywt.waverec(coeffs, wavelet='db4')
        
    if normalise:
        norm_denoised_signal = (denoised_signal - denoised_signal.min())/(denoised_signal.max()-denoised_signal.min())
        return norm_denoised_signal
    return denoised_signal


def denoise_signals(signals, DWT_state, butterworth_state, normalise_state):
    #denoising the signals through desired method
    if DWT_state:
        print('denoising signals through Discrete Wavelet Transform')
    elif butterworth_state:
        print('denoising signals through butterworth filter method')
    if normalise_state:
        print('normalising signals')
    denoised_signals = np.zeros(shape=(signals.shape))
    for i, signal in enumerate(signals):
        for j in range(0, len(signals[0, :])):
            denoised_signals[i][j] = denoise(signal[j], DWT=DWT_state, normalise=normalise_state, butter = butterworth_state)
    return denoised_signals