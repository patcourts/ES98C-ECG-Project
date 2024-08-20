from database.data import Data

#used for cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#used to encoded strings to binary 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#used for class weighting
from sklearn.utils import class_weight

import numpy as np



train_splits = {}
train_splits['train'] = 0.7
train_splits['test'] = 0.3



def get_data_CNN(splits, denoise_method):
    #train spit shouldnt be in here because takes ages to do first bits....

    #creating DATA object
    ptb_binary_NN = Data(database = 'ptbdb', denoise_method=denoise_method, estimation_method = 'NN', train_splits=splits, binary = True, parameterisation = False)

    ptb_binary_NN.run()

    if splits is None:
        return ptb_binary_NN.input_data, ptb_binary_NN.labels
        
    else:
        return ptb_binary_NN.split_data

def time_shift(data, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    augmented_data = np.roll(data, shift)
    return augmented_data

def scale_data(data, scale_factor=0.1):
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
    augmented_data = data * scale
    return augmented_data

def augment_data(data):
    augmented_data = []
    for signal in data:
        augmented_data.append(signal)
        augmented_data.append(time_shift(signal))
        augmented_data.append(scale_data(signal))
    return np.array(augmented_data)

def convert_data_to_CNN_format(data, channel_indices):
    X_train_all = []
    X_test_all = []
    y_train_all = []
    y_test_all = []

    if channel_indices == 'all':
        channel_indices = [0, 1, 2, 3, 4, 5]

    #accessing all data in desired channels
    for i in channel_indices:
        X_train_all.append(data[i]['X_train'])
        y_train_all.append(data[i]['y_train'])

        X_test_all.append(data[i]['X_test'])
        y_test_all.append(data[i]['y_test'])

    X_train_all_list = [element for array in X_train_all for element in array]
    y_train_all_list = [element for array in y_train_all for element in array]
    X_test_all_list = [element for array in X_test_all for element in array]
    y_test_all_list = [element for array in y_test_all for element in array]

    #encoding y labels for CNN
    # label_encoder = LabelEncoder()
    # y_train_all_encoded = label_encoder.fit_transform(y_train_all_list)
    # y_test_all_encoded = label_encoder.transform(y_test_all_list)
    y_train_all_encoded = [0 if label == 'Unhealthy' else 1 for label in y_train_all_list]
    y_test_all_encoded = [0 if label == 'Unhealthy' else 1 for label in y_test_all_list]

    #changing into format appropriate for CNN
    #CNN needs to know number of features per sample, i.e. 1D
    X_train_all_cnn = np.expand_dims(X_train_all_list, axis=-1) 
    y_train_all_cnn = np.expand_dims(y_train_all_encoded, axis=-1) 
    X_test_all_cnn = np.expand_dims(X_test_all_list, axis=-1) 
    y_test_all_cnn = np.expand_dims(y_test_all_encoded, axis=-1) 

    #creating new dictionary
    converted_data = {}
    converted_data['X_train'] = X_train_all_cnn
    converted_data['y_train'] = y_train_all_cnn
    converted_data['y_test'] = y_test_all_cnn
    converted_data['X_test'] = X_test_all_cnn

    #finding class weight due to inbalanced class system
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_all_encoded), y=y_train_all_encoded)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    return converted_data, y_test_all_list, class_weights_dict


def make_model(depth, filters, k, print_summary = True):
    # initialise model
    cnn = Sequential()

    for _ in range(depth):
        cnn.add(Conv1D(filters=filters, kernel_size=k, padding='same', input_shape = (60000, 1)))
        cnn.add(BatchNormalization())
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling1D(pool_size=2))  # takes max out of every two
        cnn.add(Dropout(0.5))

    cnn.add(Flatten())
    cnn.add(Dense(1, activation='sigmoid'))  # use 'sigmoid' for binary classification

    # compile model
    cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy']) #binary cross entropy for binary classification

    if print_summary:
        print(cnn.summary())
        
    return cnn


def train_model(model, data, class_weights, EPOCHS, BATCH_SIZE):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(data['X_test'], data['y_test']),
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    return history, model


def get_split_data(split_percents, data, labels):
        split_data_dict = {}
        train_split = split_percents['train']
        test_split = split_percents['test']
       # val_split = split_percents['val']
        if train_split + test_split != 1:
            print('percentages dont total 1')
            return None
        else:
            print('splitting data into test and train')
            
            for i in range(0, 6):
                split_data = {}
                X_train, X_test, y_train, y_test = train_test_split(data[i], labels[i], test_size=test_split, stratify=labels[i])
                split_data['X_train'] = X_train
                split_data['X_test'] = X_test
                split_data['y_train'] = y_train
                split_data['y_test'] = y_test
                split_data_dict[i] = split_data

            return split_data_dict
