import wfdb
import numpy as np
from tqdm import tqdm

from filter import filter_database
from preprocessing import denoise_signals
from utils import get_sig_array_from_patients, get_nan_indices, convert_multi_dict_to_array, convert_multi_dict_to_array1
from parameterisation import get_all_params
from feature_selection import select_features
from binary_classification import get_best_estimators, get_scores_and_probs
from scoring_metrics import scoring_function, print_scores_for_channel

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
health_state = np.array(allowed_patients.get_diagnoses())

#access signals from the allowed patients
signals = get_sig_array_from_patients(allowed_patients)

#denoising the signals through desired method
denoised_signals = denoise_signals(signals, DWT_state = True, butterworth_state = False, normalise_state=True)

#finding signals that have been removed due to poor quality
nan_indices = get_nan_indices(denoised_signals)

#getting all parameters
params = get_all_params(denoised_signals, allowed_patients, nan_indices)

#converting parameter dictionary to array for easy investigation
params_array, no_feats = convert_multi_dict_to_array(params, nan_indices, health_state)

#setting the desired number of features to be used in ML model
desired_no_feats = 4

#chosing selected features
selected_features = select_features(params, params_array, health_state, nan_indices, desired_no_feats)

for i in range(0, no_channels):
    print(f"Selected features for channel {i}:")
    print(list(selected_features[i].keys()))

#converting to array for use in ML
selected_features_array, no_features = convert_multi_dict_to_array1(selected_features, nan_indices, health_state)


# define hyperparameter grid to test
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale']#including 'auto' aswell takes forever
}

#find the best set of hyperparameters for each channel, tuned on the desired scoring function
best_estimators = get_best_estimators(selected_features_array, param_grid, scoring_function, health_state, nan_indices)

#perform skfold to get scores for each channel as well as their probabilities
score, bal_acc, probabilities, sample_percentages, y_tests, test_indices = get_scores_and_probs(selected_features_array, health_state, nan_indices, best_estimators, scoring_function, n_splits=3)

#printing scores for each channel
print_scores_for_channel(score, bal_acc)