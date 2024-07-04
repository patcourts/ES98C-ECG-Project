import wfdb
import numpy as np
from tqdm import tqdm

from filter import filter_database
from preprocessing import denoise_signals
from utils import get_sig_array_from_patients, get_nan_indices
from parameterisation import get_all_params


#creates a list containing the directories of all the ECGs within the data base
patients = wfdb.get_record_list('ptbdb') 

#Stating which filters too apply
duplicate_data_filtering = True
desired_sample_length = 60000
age_data_filtering = True
smoking_data_filtering = False #smoking filter gets rid of alot of healthy patients -leaves only 3 remaining
gender_data_filtering = True

#156 people and 3 healthy if all filters applied

#applying filters to the data base
allowed_patients = filter_database(patients, desired_sample_length, duplicate_data_filtering, age_data_filtering, smoking_data_filtering, gender_data_filtering)
no_patients = allowed_patients.count_patients()
sample_length = allowed_patients.get_patients(0).get_desired_sample_length()
channel_list = allowed_patients.get_patients(0).get_channel_names()
no_channels = len(channel_list)

#access signals from the allowed patients
signals = get_sig_array_from_patients(allowed_patients)

#denoising the signals through desired method
denoised_signals = denoise_signals(signals, DWT_state = True, butterworth_state = False, normalise_state=True)

#finding signals that have been removed due to poor quality
nan_indices = get_nan_indices(signals)

#getting all parameters
params = get_all_params(signals, allowed_patients, nan_indices)



