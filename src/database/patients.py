import wfdb
import numpy as np
import matplotlib.pyplot as plt


class Patient:
    def __init__(self, name, age, gender, health_state, diagnosis, smoker):
        self._docstring = str
        self._name = str
        self._age = int
        self._gender = str
        self._health_state = str
        self._diagnosis = str
        self._smoker = str
        self._ecg_data = None
        self._signal = None
        self._desired_sample_length = int
        self._channel_names = ["v1", "v2", "v3", "v4", "v5", "v6"]
        
    def set_docstring(self, docstr):
        self._docstring = docstr

    def set_name(self, name):
        self._name = name
        
    def set_health_state(self, health_state):
        self._health_state = health_state

    def set_diagnosis(self, diagnosis):
        self._diagnosis = diagnosis

    def set_ecg_data(self, ecg_data):
        self._ecg_data = ecg_data
        
    def set_signal(self, signal):
        self._signal = signal

    def set_age(self, age):
        self._age = age
        
    def set_gender(self, gender):
        self._gender = gender
    
    def set_smoker_status(self, smoker):
        self._smoker = smoker
        
    def set_channel_names(self, channel_names):
        self._channel_names = channel_names
    
    def set_desired_sample_length(self, desired_sample_length):
        self._desired_sample_length = desired_sample_length
        
    def get_docstring(self):
        return self._docstring
    
    def get_name(self):
        return self._name
    
    def get_age(self):
        return self._age
    
    def get_gender(self):
        return self._gender
    
    def get_smoker_status(self):
        return self._smoker
    
    def get_ecg_data(self):
        return self._ecg_data
    
    def get_diagnosis(self):
        return self._diagnosis   
    
    def get_health_state(self):
        return self._health_state
    
    def get_channel_names(self):
        return self._channel_names
    
    def get_desired_sample_length(self):
        return self._desired_sample_length
    
    def get_info(self):
        info = f"Name: {self._name} \n Age: {self._age}\n Gender: {self._gender}\n"
        info += f"Health State: {self._health_state}\nDiagnosis: {self._diagnosis}\n"
        info += f"ECG Data: {self._ecg_data} \nSmoker: {self._smoker}"
        return info
    
    def plot_ecg(self):
        wfdb.plot_wfdb(record=self.get_ecg_data(), title='All Channels for '+ self._name)
        
    def plot_signal(self):
        wfdb.plot_items(signal = self._signal)
        
    def read_record(self, given_channel_names=None):
        if given_channel_names is not None:
            self.set_channel_names(given_channel_names)
        record = wfdb.rdrecord('C:\\Users\\court\\Documents\\masters\\solo project\\ES98C-ECG-Project\\ptb-diagnostic-ecg-database-1.0.0\\ptb-diagnostic-ecg-database-1.0.0\\' + self.get_docstring(), channel_names = self.get_channel_names(), sampto = self.get_desired_sample_length())
        return record
    
    def read_signal(self, given_channel_names = None):
        if given_channel_names is not None:
            self.set_channel_names(given_channel_names)
        signal, _ = wfdb.rdsamp('C:\\Users\\court\\Documents\\masters\\solo project\\ES98C-ECG-Project\\ptb-diagnostic-ecg-database-1.0.0\\ptb-diagnostic-ecg-database-1.0.0\\' + self.get_docstring(), channel_names = self.get_channel_names(), sampto = self.get_desired_sample_length())
        return signal
    
    def get_signal(self):
        return self._signal
    
    def plot_channels(self, channel_names):
        record = self.read_record(given_channel_names = channel_names)
        wfdb.plot_wfdb(record=record, title='Channel(s) '+ ', '.join(channel_names) + ' for '+ self._name)
        
        
        
class PatientCollection:
    def __init__(self):
        self._patients = []
        self._diagnosis_list = ['Myocardial infarction', 'Healthy control', 'Dysrhythmia', 'Cardiomyopathy', 'Hypertrophy', 'Bundle branch block', 'Valvular heart disease', 'Stable angina', 'Myocarditis']


    def add_patient(self, patient):
        self._patients.append(patient)
        
        
    def get_patients(self, indx):
        return self._patients[indx] 
    
    
    def get_records(self):
        return [patient.get_ecg_data() for patient in self._patients]
    
    def count_patients(self):
        return len(self._patients)

    def get_all_info(self):
        return [patient.get_info() for patient in self._patients]

    def count_smokers(self):
        return sum(1 for patient in self._patients if patient._smoker == 'yes')
    
    def count_healthy(self):
        return sum(1 for patient in self._patients if patient._health_state == 'Healthy')
    
    def count_unhealthy(self):
        return sum(1 for patient in self._patients if patient._health_state == 'Unhealthy')
    
    def count_gender(self):
        return sum(1 for patient in self._patients if patient._gender == 'Male')
    
    def get_diagnosis_list(self):
        return self._diagnosis_list
    
    def get_diagnoses(self):
        return [patient._health_state for patient in self._patients]
    
    def get_diagnosis_counts(self):
        diagnoses = [patient._diagnosis for patient in self._patients] #creates list of each diagnosis within the patients in the list
        diagnosis_counts = {diagnosis: diagnoses.count(diagnosis) for diagnosis in set(diagnoses)} #counts each diagnosis, set()removes repeats
        return diagnosis_counts
    
    def plot_diagnosis(self):
        diagnosis_counts = self.get_diagnosis_counts()
        fig, axes = plt.subplots(1, 1, figsize = (12, 10))
        
        bars = axes.bar(diagnosis_counts.keys(), diagnosis_counts.values())
        
        # annotate the values on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom')
        axes.set_xlabel('Diagnosis')
        axes.set_ylabel('Count')
        axes.set_title('Diagnosis Distribution')
        axes.legend([f"Total Number of People: {self.count_patients()}, Healthy: {self.count_healthy()}", f"Unhealthy: {self.count_unhealthy()}"])
        plt.show()
        
    def plot_sample_ecg(self):
        plot_indx = np.random.randint(0, self.count_patients())
        patient = self.get_patient(plot_indx)
        patient.plot_ecg()
        
    
        
        