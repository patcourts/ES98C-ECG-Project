from database.patients import Patient, PatientCollection
import wfdb
from tqdm import tqdm
import neurokit2 as nk


def filter_database(patients, sample_length:int, duplicate_data_filtering:bool, age_data_filtering:bool, smoking_data_filtering:bool, gender_data_filtering:bool):
    #indexes from the database metadata
    age_meta_idx = 0
    smoking_meta_idx = 8
    gender_meta_idx = 1
    diagnosis_meta_idx = 4
    previous_label = 'n/a'

    allowed_patients = PatientCollection()

    print('Filtering Database')

    j=0
    for i in tqdm(range(0, len(patients))):
        #creating new patient instance
        new_Patient = Patient(None, None, None, None, None, None)
        new_Patient.set_desired_sample_length(sample_length)
        
        #setting bool to true unless conditions for appending is not met
        append = True
        
        #try and except clause to ensure only ECGs longer than the minimum length are kept
        try:
            record = wfdb.rdrecord('C:\\Users\\court\\Documents\\masters\\solo project\\ES98C-ECG-Project\\ptb-diagnostic-ecg-database-1.0.0\\ptb-diagnostic-ecg-database-1.0.0\\' + patients[i], channel_names = new_Patient.get_channel_names(), sampto = new_Patient.get_desired_sample_length())
            signals, fields = wfdb.rdsamp('C:\\Users\\court\\Documents\\masters\\solo project\\ES98C-ECG-Project\\ptb-diagnostic-ecg-database-1.0.0\\ptb-diagnostic-ecg-database-1.0.0\\' + patients[i], sampto = new_Patient.get_desired_sample_length(), channel_names = new_Patient.get_channel_names())
            
            #append ECG data to Patient instance
            new_Patient.set_ecg_data(record)
            
            new_Patient.set_docstring(patients[i])
            
            #labelling the diagnosis within the class
            if record.comments[diagnosis_meta_idx][22:] not in allowed_patients.get_diagnosis_list():
                append = False
            else:
                new_Patient.set_diagnosis(record.comments[diagnosis_meta_idx][22:])
                
            #labelling the health state of each patient class
            if record.comments[diagnosis_meta_idx][22:] == 'Healthy control':
                new_Patient.set_health_state('Healthy')
            else:
                new_Patient.set_health_state('Unhealthy')
            
            if age_data_filtering:
                #making sure age meta data is available
                if record.comments[age_meta_idx] == None:
                    append = False
                else:
                    new_Patient.set_age(int(record.comments[age_meta_idx][5:]))
                    
            if smoking_data_filtering:
                #ensuring smoking meta data is available
                if record.comments[smoking_meta_idx][8:] not in ('yes', 'no'):
                    append = False
                else:
                    new_Patient.set_smoker_status(record.comments[smoking_meta_idx][8:])
                    
            if gender_data_filtering:
                #ensuring gender meta data is available
                if record.comments[gender_meta_idx][5:] not in ('male', 'female'):
                    append = False
                else:
                    new_Patient.set_gender(record.comments[gender_meta_idx][5:])
                
            if duplicate_data_filtering:
                #only appending the first ECG from each patient to remove duplicates
                patient_number = patients[i][:10]
                if patient_number == previous_label:
                    append = False
                else:
                    previous_label = patient_number
                    new_Patient.set_name(patients[i][:10])
                        
            if append:
                
    #             #removes whole patient if signal quality of one channel is not good enough     
    #             signal_quality = []
    #             bad_signals = []
    #             for j in range(0, len(new_Patient.get_channel_names())):
    #                     cleaned_signal = nk.ecg_clean(signals[:, j])
    #                     quality = nk.ecg_quality(cleaned_signal, method='zhao2018')
    #                     signal_quality.append(quality)
    #             [bad_signals.append(i) for x in signal_quality if x == 'Unacceptable']

    #             if len(bad_signals) == 0:
    #                 allowed_patients.add_patient(new_Patient)
                    
                #removing signals from specific channels that are not of good quality        
                bad_signal_channel = []
                for j in range(0, len(new_Patient.get_channel_names())):
                        cleaned_signal = nk.ecg_clean(signals[:, j])
                        quality = nk.ecg_quality(cleaned_signal, method='zhao2018')
                        if quality == 'Unacceptable':
                            bad_signal_channel.append(j)
                            signals[:, j] = None
                
                new_Patient.set_signal(signals)
                
                #appending patients as long as at least one channel contains signal of accpetable quality
                if len(bad_signal_channel) != 6:
                    allowed_patients.add_patient(new_Patient)
            
        except ValueError as ve:
            None
            #print(f"Error reading record for patient {patients[i]}: {ve}")

    return allowed_patients