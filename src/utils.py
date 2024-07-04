import numpy as np
from tqdm import tqdm
import neurokit2 as nk
from patients import PatientCollection

def get_sig_array_from_patients(patients: PatientCollection, invert_signals=False):
    no_patients = patients.count_patients()
    sample_length = patients.get_patients(0).get_desired_sample_length()
    channel_list = patients.get_patients(0).get_channel_names()
    no_channels = len(channel_list)

    signals = np.zeros(shape=(no_patients, no_channels, sample_length))
    for i in tqdm(range(0, no_patients)):
        #looping over each channel investigated
        channels = np.zeros(shape=(no_channels, sample_length))
        for j, channel in enumerate(channel_list):
            signal = patients.get_patients(i).get_signal()[:, j]
           
    
            #flips signals if in need of inverting
            if invert_signals:
                if not np.isnan(signal).all():
                    inversion_corrected_signal = nk.ecg_invert(signal)[0]
                    channels[j] = inversion_corrected_signal
                else:
                    channels[j] = signal
            else:
                channels[j] = signal

        signals[i] = channels
    return signals

def parameter_averages(parameter, health_state):
    """
    calculates healthy and unhealthy means and std of parameter for easy comparisson
    returns np.array containing: healthy mean, healthy std, unhealthy mean, unhealhty std
    """
    encoded_health_state = [True if label == 'Unhealthy' else False for label in health_state]

    unhealthy_param = parameter[np.array(encoded_health_state)]
    healthy_param = parameter[~np.array(encoded_health_state)]
    
    unhealthy_param_av = np.mean(unhealthy_param)
    unhealthy_param_std = np.std(unhealthy_param)
    
    healthy_param_av = np.mean(healthy_param)
    healthy_param_std = np.std(healthy_param)
    
    return np.array([healthy_param_av, healthy_param_std, unhealthy_param_av, unhealthy_param_std])


def print_averages(parameter, parameter_name, health_state):
    """
    print healthy and unhealthy means of parameter in nice format
    """
    
    means = parameter_averages(parameter, health_state)
    
    print(f"Unhealthy {parameter_name}: mean:{means[2]}, std: {means[3]}")
    print(f"Healthy {parameter_name}: mean:{means[0]}, std:{means[1]}")
    return None

def get_nan_indices(signals):
    nan_indices = []
    for j in range(0, len(signals[0, :])):
        signal_nan_indices = []
        for i, signal in enumerate(signals[:, j]):
            if np.isnan(signal).all():
                signal_nan_indices.append(False)
            else:
                signal_nan_indices.append(True)
        nan_indices.append(signal_nan_indices)

    nan_indices = np.array(nan_indices)
    return nan_indices