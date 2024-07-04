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

def get_time_params(signals):
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



    return rr_means_list, rr_stds_list, rr_RMSSD_list, rr_pNN50s_list, rr_amps_list, means_list, stds_list, skews_list, kurtosiss_list


def power_bands(signal):
    """
    performs welch transform to calculate the power within high and low frequency bands
    as well as the total power within the frequency spectrum
    """
    fs = 1000 #Hz, sampling frequency
    fs_interpolate = 3 #Hz, arbitrary value, 3 is typically used
    
    # define frequency bands, based on typical values used in HRV
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.40)
    
    #gets rpeaks and amplitudes
    peaks, amps = get_peaks(signal)
    
    # calculates rr intervals in seconds
    rr_intervals = np.diff(peaks)/fs
    
    # time points of the RR intervals
    rr_times = np.cumsum(rr_intervals)
    rr_times = np.insert(rr_times, 0, 0)  # add time zero at the beginning

    # interpolation for frequency transform
    interpolated_time = np.arange(0, rr_times[-2], 1/fs_interpolate)
    interpolated_rr = interp1d(rr_times[:-1], rr_intervals, kind='cubic')(interpolated_time)

    #welch spectrum
    f, psd = welch(interpolated_rr, fs=fs_interpolate, nperseg=128)
    
    # integrate the power spectral density over the frequency bands
    lf_power = np.trapz(psd[(f >= lf_band[0]) & (f <= lf_band[1])], f[(f >= lf_band[0]) & (f <= lf_band[1])])
    hf_power = np.trapz(psd[(f >= hf_band[0]) & (f <= hf_band[1])], f[(f >= hf_band[0]) & (f <= hf_band[1])])
    
    #total power, integrated over whole range
    total_power = np.trapz(psd, f)
    
    return lf_power, hf_power, lf_power/hf_power, total_power

def get_frequency_params(signals):
    lfs_list = []
    hfs_list = []
    ratios_list = []
    tps_list = []

    for j in range(0, len(signals[0, :])):
        lfs = []
        hfs = []
        ratios = []
        tps = []
        
        for signal in signals[:, j]:
            if not np.isnan(signal).all():
                lf, hf, ratio, tp = power_bands(signal)
                lfs.append(lf)
                hfs.append(hf)
                ratios.append(ratio)
                tps.append(tp)
                
    lfs_list.append(lfs)
    hfs_list.append(hfs)
    ratios_list.append(ratios)
    tps_list.append(tps)

    return lfs_list, hfs_list, ratios_list, tps_list


def calculate_poincare_sd(sig, remove_outliers=False, use_nk=False):
    #get rr intervals
    rr_intervals = get_rri(sig)[0]
    
    if remove_outliers:
        rr_intervals = rr_intervals[outliers_indices_z_score(rr_intervals)]
        
    if use_nk:
        peaks, info = nk.ecg_peaks(sig, sampling_rate=1000)
        nl_indices = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
        return nl_indices['HRV_SD1'], nl_indices['HRV_SD2'], nl_indices['HRV_SD1SD2']
        
    #separating into subsequent coordinates for Poincaré plot
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    #calculating SD1, perpendicular to y=x
    diff_rr = np.array(rr_n) - np.array(rr_n1)
    sd1 = np.sqrt(np.var(diff_rr/np.sqrt(2)))
    
    # calculating SD2, along y=x
    sum_rr = rr_n + rr_n1
    sd2 = np.sqrt(np.var(sum_rr/np.sqrt(2)))
    
    # calculating ratio
    sd_ratio = sd2/sd1
   
    if remove_outliers:
        #counting intervals outside SD1 and SD2
        count_outside_sd1 = np.sum(np.abs(diff_rr / np.sqrt(2)) > sd1)
        count_outside_sd2 = np.sum(np.abs(sum_rr / np.sqrt(2)) > sd2)

        out = count_outside_sd1 + count_outside_sd2
        return sd1, sd2, sd_ratio, count_outside_sd1, count_outside_sd2, out
    
    else:
        return sd1, sd2, sd_ratio
    

def calculate_shannon_entropy(signal, num_bins=500, use_nk=False):
    rr_intervals = get_rri(signal)[0]
    
    if use_nk:
        peaks, info = nk.ecg_peaks(signal, sampling_rate=1000)
        nl_indices = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
        return nl_indices['HRV_ShanEn']
    
    # discretize RR intervals into bins and calculate probabilities
    hist, bin_edges = np.histogram(rr_intervals, bins=num_bins, density=True)
    
    probabilities = hist / np.sum(hist)
    
    # Calculate Shannon Entropy using equation
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))  # adding a small value to avoid log(0)
    
    return shannon_entropy


def calculate_sample_entropy(signal, m=2, r=0.2):
    rr_intervals = get_rri(signal)[0]
    N = len(rr_intervals)
    r *= np.std(rr_intervals)  # tolerance r is usually set as a fraction of the standard deviation
    
    def _phi(m):
        X = np.array([rr_intervals[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    
    return -np.log((_phi(m + 1)) / (_phi(m)))


def get_nonlinear_params(signals):
    shannon_ens_list = []
    sd1s_list = []
    sd2s_list = []
    sd_ratios_list = []
    samp_ens_list = []

    for j in range(0, len(signals[0, :])):
        sd1s = []
        sd2s = []
        sd_ratios = []
        shannon_en = []
        samp_en = []

        for signal in signals[:, j]:
            if not np.isnan(signal).all():
                sd1, sd2, sd_ratio = calculate_poincare_sd(signal, remove_outliers=False)
                sd1s.append(sd1)
                sd2s.append(sd2)
                sd_ratios.append(sd_ratio)

                shannon_en.append(calculate_shannon_entropy(signal))
                
                samp_en.append(calculate_sample_entropy(signal))

            
        sd1s_list.append(sd1s)
        sd2s_list.append(sd2s)
        sd_ratios_list.append(sd_ratios)
        shannon_ens_list.append(shannon_en)
        samp_ens_list.append(samp_en)
    
    return sd1s_list, sd2s_list, sd_ratios_list, shannon_ens_list, samp_ens_list

def get_covariates(patients):
    no_patients = patients.count_patients()
    ages=np.zeros(no_patients)
    for i in range(0, no_patients):
        ages[i] = patients.get_patients(i).get_age()
    return ages
    


def get_all_params(signals, patients, nan_indices):
    print('calculating time domain parameters')
    rr_means, rr_stds, rr_RMSSD, rr_pNN50s, rr_amps, means, stds, skews, kurtosiss = get_time_params(signals)
    print('calculating frequency domain parameters')
    lfs, hfs, ratios, tps = get_frequency_params(signals)
    print('calculating non linear domain parameters')
    sd1s, sd2s, sd_ratios, shannon_ens, samp_ens = get_nonlinear_params(signals)
    
    

    ages = get_covariates(patients)
    ages_list = []
    for j in range(0, len(signals[0, :])):
        ages_list.append(ages[nan_indices[j]])

    params = {}

    #time components
    params['rr_mean'] = rr_means
    params['rr_std'] = rr_stds
    params['rr_amps'] = rr_amps
    params['RMSSD'] = rr_RMSSD
    params['pNN50'] = rr_pNN50s

    #params['QRS_mean'] = QRS_means_list #contains nan so need to remove those so can use the filtering methods
    #params['QRS_std'] = QRS_stds_list

    params['mean'] = means
    params['std'] = stds
    params['skews'] = skews
    params['kurtosis'] = kurtosiss

    #frequency components
    params['hf'] = hfs
    params['lf'] = lfs
    params['power_ratio'] = ratios
    params['total_power'] = tps

    #nonlinear components
    params['shannon_en'] = shannon_ens
    #params['sample_en'] = samp_ens need to do imputing if want to include it, due to some nan values
    params['sd_ratio'] = sd_ratios
    params['sd1'] = sd1s
    params['sd2'] = sd2s
    #params['nk_sd_ratio'] = nk_sd_ratios_list
    #params['sd_ratio_outliers_removed'] = sd_ratios_outliers_removed_list
    #params['fd'] = fd_list

    #covariates
    params['age'] = ages_list

    return params
