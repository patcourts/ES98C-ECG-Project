# FSL model

from database.data import Data
from database.filter import filter_database
import wfdb
import numpy as np
from database.utils import get_sig_array_from_patients


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


channel_one_sig = signals[0, :]


#getting categorical diagnosis for each patient
detailed_health_state = []
for i in range(0, no_patients):
    detailed_health_state.append(allowed_patients.get_patients(i).get_diagnosis())

detailed_health_state = np.array(detailed_health_state)

#these have very few appearances in data set so are removed
diagnoses_to_remove = ['Stable angina', 'Myocarditis', 'Valvular heart disease', 'Hypertrophy']

#finding indices with which to remove from the dataset
diagnosis_indices = []
new_health_state = []
for j in range(0, no_channels):
    filter_diagnosis = []
    for diagnosis in detailed_health_state[nan_indices[j]]:
        if diagnosis in diagnoses_to_remove:
            filter_diagnosis.append(False)
        else:
            filter_diagnosis.append(True)
    diagnosis_indices.append(filter_diagnosis)
    new_health_state.append(detailed_health_state[nan_indices[j]][diagnosis_indices[j]])





# #gonna start with just one channel
# input_data = denoised_signals[0, :][nan_indices[0]][diagnosis_indices[j]]
# labels = new_health_state[0]

# print(len(input_data))
# print(len(labels))