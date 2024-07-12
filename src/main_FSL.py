# FSL model

from database.data import Data
from database.filter import filter_database
import wfdb
import numpy as np
from database.utils import get_sig_array_from_patients

train_splits = {}
train_splits['train'] = 0.7
train_splits['test'] = 0.3

#creating DATA object
ptb = Data(database = 'ptbdb', denoise_method='DWT', train_splits=train_splits, binary = False, parameterisation = False)

ptb.run()

print(ptb.labels)
print(ptb.split_data)

print(ptb.split_data[0]['X_train'])
print(ptb.split_data[5]['y_test'])
print(len(ptb.split_data[4]['X_test']))
