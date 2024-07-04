from scipy.signal import correlate #for autocorrelation for signal embedding
from scipy.stats import skew, kurtosis #for signal measures
from scipy.signal import welch #for frequency band
from scipy.interpolate import interp1d #for interpolation of signal
import neurokit2 as nk #for peak finding, and other measures
from pyhrv.hrv import hrv #think neurokit might use this, needed otherwise issues
import numpy as np


def outliers_indices_z_score(data, threshold=2):
    """
    finds outliers from data, those outside the threshold region
    return the indices of the data to be kept
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    indices = np.argwhere([abs(z)<threshold for z in z_scores])
    return indices

def get_peaks(sig):
    """
    returns the location of the peaks (calculated through neurokit) and their average amplitude
    """
    #using neurokit to find the location of the r peaks
    peak_dict, info = nk.ecg_peaks(sig, sampling_rate=1000)
    peaks = info['ECG_R_Peaks']
    
    #calculating amplitude of R peak
    peak_amp = sig[peaks] - np.median(sig) #median used as baseline
    peak_amp_av = np.mean(peak_amp) 
    return peaks, peak_amp_av
        

def get_rri(sig, remove_outliers=False):
    """
    calculates the rr intervals and peak amplitudes of a signal
    has option to remove outliers from the rr interval array
    """
    #gets rpeaks and amplitudes
    peaks, amps = get_peaks(sig)
    
    # calculates rr intervals
    rri = np.diff(peaks)
    
    #removal of outliers
    if remove_outliers:
        rri_outlier_indices = outliers_indices_z_score(rri).reshape(-1)
        rr_intervals = rri[rri_outlier_indices]

    else:
        rr_intervals = rri
    return rr_intervals, amps

def RR_analysis(signal, remove_outliers=False):
    """
    calculates parameters based on the RR intervals of the signal
    returns mean of RR intervals, std of RR intervals, RMSSD, pNN50, amplitude of r peaks
    """
    peak_distances, amps = get_rri(signal, remove_outliers=remove_outliers)
    
    # stastitical analysis on rr intervals
    mean_RR = np.mean(peak_distances)
    std_RR = np.std(peak_distances)

    #RMSSD
    # computes differences between successive RR intervals for RMSSD
    diff_RR_intervals = np.diff(peak_distances)
    
    RMSSD_RR = np.sqrt(np.mean(diff_RR_intervals**2))
    
    #pNN50
    # the number of successive RR intervals that differ by more than 50 ms 
    NN50 = np.sum(np.abs(diff_RR_intervals) > 50)
    # divide by total number of RR intervals
    pNN50 = (NN50 / len(peak_distances)) * 100
    
    
    return mean_RR, std_RR, RMSSD_RR, pNN50, amps


def QRS_complex(signal):
    #required nk cleaned signal
    cleaned_signal = nk.ecg_clean(signal, sampling_rate=1000)
    
    #calculates location of all the various peaks in the signal
    wave_dict, signals_df = nk.ecg_delineate(signal)
        
    q_peaks = signals_df['ECG_Q_Peaks']
    s_peaks = signals_df['ECG_S_Peaks']
        
    #calculating duration of QRS complex
    qrs = np.array(s_peaks) - np.array(q_peaks)

    #calculating mean and std
    qrs_mean = np.nanmean(qrs)
    qrs_std = np.nanstd(qrs)
    
    return qrs_mean, qrs_std


def get_moments(signal):
    """
    returns the moments of the signal up to the fourth order
    """
    mean = np.mean(signal)
    std = np.std(signal)
    skew_ecg = skew(signal)
    kurtosis_ecg = kurtosis(signal)
    return mean, std, skew_ecg, kurtosis_ecg

def get_time_domain_params(signals):
    means_list = []
    stds_list = []
    skews_list = []
    kurtosiss_list = []

    rr_means_list = []
    rr_stds_list = []
    rr_RMSSD_list = []
    rr_pNN50s_list = []
    rr_amps_list = []

    for j in range(0, len(signals[0, :])):
        means = []
        stds = []
        skews = []
        kurtosiss = []

        rr_means = []
        rr_stds = []
        rr_RMSSD = []
        rr_pNN50s = []
        rr_amps = []

    
        for signal in signals[:, j]:
            if not np.isnan(signal).all():

                mean, std, ecg_skew, ecg_kurtosis = get_moments(signal)
                rr_mean, rr_std, rr_rmssd, rr_pNN50, rr_amp = RR_analysis(signal, remove_outliers=True)

                means.append(mean)
                stds.append(std)
                skews.append(ecg_skew)
                kurtosiss.append(ecg_kurtosis)

                rr_means.append(rr_mean)
                rr_stds.append(rr_std)
                rr_RMSSD.append(rr_rmssd)
                rr_pNN50s.append(rr_pNN50)
                rr_amps.append(rr_amp)
            

        rr_means_list.append(rr_means)
        rr_stds_list.append(rr_stds)
        rr_RMSSD_list.append(rr_RMSSD)
        rr_pNN50s_list.append(rr_pNN50s)
        rr_amps_list.append(rr_amps) 

        means_list.append(means)
        stds_list.append(stds)
        skews_list.append(skews)
        kurtosiss_list.append(kurtosiss) 



    return rr_means_list, rr_stds_list, rr_RMSSD_list, rr_pNN50, rr_amps_list, means_list, stds_list, skews_list, kurtosiss_list
