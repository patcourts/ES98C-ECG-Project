import numpy as np
from tqdm import tqdm
import neurokit2 as nk
from database.patients import PatientCollection

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


def convert_multi_dict_to_array(params, nan_indices, health_state):
    no_features = len(params)
    params_list = []
    for j in range(0, len(nan_indices)):
        params_array = np.zeros(shape=(len(health_state[nan_indices[j]]), no_features))
        for i, values in enumerate(params.values()):
            params_array[:, i] = values[j]
        params_list.append(params_array)
    return params_list, no_features

def convert_single_dict_to_array(params, health_state):
    no_features = len(params)
    params_array = np.zeros((len(health_state), no_features))
    for i, values in enumerate(params.values()):
        params_array[:, i] = values
    return params_array, no_features

def reduce_parameter_set(params, selected_indices, channel):
    select_params = {}
    for i, key in enumerate(params.keys()):
        if i in selected_indices:
            select_params[key] = params[key][channel]
    return select_params
        
def reduce_parameter_set_single(params, selected_indices):
    select_params = {}
    for i, key in enumerate(params.keys()):
        if i in selected_indices:
            select_params[key] = params[key]
    return select_params

def convert_multi_dict_to_array1(params_dict, nan_indices, health_state):
    no_features = len(params_dict[0])
    params_list = []
    for j in range(0, len(nan_indices)):
        params_array = np.zeros((len(health_state[nan_indices[j]]), no_features))
        for i, values in enumerate(params_dict[j].values()):
            params_array[:, i] = values
        params_list.append(params_array)
    return params_list, no_features

def reconstruct_probs(probs, test_indices, nan_indices, no_patients, n_splits):
    reconstructed_probs = np.zeros(no_patients)
    reconstructed_probs[~nan_indices] = np.nan
    allowed_indices = np.arange(0, no_patients)[nan_indices]

    for z in range(0, n_splits):
        for i, indice in enumerate(test_indices[z]):
            reconstructed_probs[allowed_indices[indice]] = probs[z][i][1]
            
    return reconstructed_probs


def get_reconstructed_probabilities(probs, test_indices, nan_indices, no_patients, n_splits):
    reconstructed_probs = []
    for i in range(0, len(nan_indices)):
        reconstructed_prob = reconstruct_probs(probs[i], test_indices[i], nan_indices[i], no_patients, n_splits)
        reconstructed_probs.append(reconstructed_prob)
    return reconstructed_probs