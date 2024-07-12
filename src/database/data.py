import numpy as np
from database.preprocessing import denoise_signals
from database.utils import get_sig_array_from_patients
from sklearn.model_selection import train_test_split
from database.patients import Patient, PatientCollection
from database.filter import filter_database
import wfdb


class Data:

    def __init__(self, database: str, denoise_method: str, train_splits: dict, binary: bool, parameterisation: bool):
        #determining methods to be used
        self.patients = wfdb.get_record_list(database)
        self.denoise_method = denoise_method
        self.binary = binary
        self.parameterisation = parameterisation

        #for managig and processing database
        self.allowed_patients = PatientCollection
        self.signals = np.ndarray
        self.nan_indices = np.ndarray
        self.denoised_signals = np.ndarray
        self.diagnosis_indices = np.ndarray

        #for use in ML models
        self.train_splits = train_splits
        self.labels = list[str]
        self.input_data = np.ndarray
        self.split_data = dict


    def filter_database(self):
        #Stating which filters too apply
        duplicate_data_filtering = True
        desired_sample_length = 60000
        age_data_filtering = True
        smoking_data_filtering = False #smoking filter gets rid of alot of healthy patients -leaves only 3 remaining
        gender_data_filtering = True
        
        #applying filters to the data base
        self.allowed_patients = filter_database(self.patients, desired_sample_length, duplicate_data_filtering, age_data_filtering, smoking_data_filtering, gender_data_filtering)
        return self.allowed_patients

    def get_signals(self):
        self.signals = get_sig_array_from_patients(self.allowed_patients)

    def get_nan_indices(self):
        nan_indices = []
        for j in range(0, len(self.signals[0, :])):
            signal_nan_indices = []
            for i, signal in enumerate(self.signals[:, j]):
                if np.isnan(signal).all():
                    signal_nan_indices.append(False)
                else:
                    signal_nan_indices.append(True)
            nan_indices.append(signal_nan_indices)

        nan_indices = np.array(nan_indices)
        self.nan_indices = nan_indices
        return nan_indices
    
    def preprocess_signals(self, method: str, normalise_state):

        allowed_preprocessing_methods = ['butterworth', 'DWT']
        if method not in allowed_preprocessing_methods:
            print(f'allowed methods are: {allowed_preprocessing_methods}')
            return None
        else:
            denoised_signals = denoise_signals(self.signals, method, normalise_state)
            self.denoised_signals = denoised_signals
            return denoised_signals


    def get_multi_labels(self):
        #getting categorical diagnosis for each patient
        detailed_health_state = []
        for i in range(0, self.allowed_patients.count_patients()):
            detailed_health_state.append(self.allowed_patients.get_patients(i).get_diagnosis())

        detailed_health_state = np.array(detailed_health_state)

        #these have very few appearances in data set so are removed
        diagnoses_to_remove = ['Stable angina', 'Myocarditis', 'Valvular heart disease', 'Hypertrophy']

        #finding indices with which to remove from the dataset
        diagnosis_indices = []
        new_health_state = []
        channel_list = self.allowed_patients.get_patients(0).get_channel_names()
        for j in range(0, len(channel_list)):
            filter_diagnosis = []
            for diagnosis in detailed_health_state[self.nan_indices[j]]:
                if diagnosis in diagnoses_to_remove:
                    filter_diagnosis.append(False)
                else:
                    filter_diagnosis.append(True)
            diagnosis_indices.append(filter_diagnosis)
            new_health_state.append(detailed_health_state[self.nan_indices[j]][diagnosis_indices[j]])

        self.labels = new_health_state
        self.diagnosis_indices = diagnosis_indices
        return new_health_state


    def get_binary_labels(self):
        self.labels = np.array(self.allowed_patients.get_diagnoses())

    def set_input_data(self):
        input_data = []
        for j in range(0, 6):
            input_data.append(self.denoised_signals[:, j][self.nan_indices[j]][self.diagnosis_indices[j]])
        self.input_data = input_data

        
    def get_split_data(self, split_percents: dict):
        split_data_dict = {}
        train_split = split_percents['train']
        test_split = split_percents['test']
       # val_split = split_percents['val']
        if train_split + test_split != 1:
            print('percentages dont total 1')
            return None
        else:
            for i in range(0, 6):
                split_data = {}
                X_train, X_test, y_train, y_test = train_test_split(self.input_data[i], self.labels[i], test_size=test_split)
                split_data['X_train'] = X_train
                split_data['X_test'] = X_test
                split_data['y_train'] = y_train
                split_data['y_test'] = y_test
                split_data_dict[i] = split_data
            self.split_data = split_data_dict
            

    def run(self):
        self.filter_database()
        self.get_signals()
        self.get_nan_indices()
        self.preprocess_signals(method=self.denoise_method, normalise_state=True)

        if self.binary:
            self.get_binary_labels()
        else:
            self.get_multi_labels()
        
        if self.parameterisation:
            self.get_params()
        else:
            self.set_input_data()


        self.get_split_data(self.train_splits)

