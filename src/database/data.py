import numpy as np
from database.preprocessing import denoise_signals
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, signals, labels):
        self.signals = signals
        self.denoised_signals = np.ndarray
        self.labels = labels

    def preprocess_signals(self, method: str, normalise_state):
        allowed_preprocessing_methods = ['butterworth, DWT']
        if method not in allowed_preprocessing_methods:
            print(f'allowed methods are: {allowed_preprocessing_methods}')
            return None
        else:
            denoised_signals = denoise_signals(self.signals, method, normalise_state)
            self.denoised_signals = denoised_signals
            return denoised_signals
        
    def train_test_val_split(self, split_percents: dict):
        train_split = split_percents['train']
        test_split = split_percents['test']
       # val_split = split_percents['val']
        if train_split + test_split != 1:
            print('percentages dont total 1')
            return None
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.denoised_signals, self.labels, test_size=test_split)
            

