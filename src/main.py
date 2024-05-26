from patients import filter_database
import wfdb
import numpy as np
import matplotlib.pyplot as plt

#creates a list containing the directories of all the ECGs within the data base
patients = wfdb.get_record_list('ptbdb') 

#definite application
duplicate_data_filtering = True
sample_length = 50000


age_data_filtering = False
smoking_data_filtering = False #smoking filter gets rid of alot of healthy patients -leaves only 3 remaining
gender_data_filtering = False

#156 people and 3 healthy if all filters applied

allowed_patients = filter_database(patients, sample_length, duplicate_data_filtering, age_data_filtering, smoking_data_filtering, gender_data_filtering)