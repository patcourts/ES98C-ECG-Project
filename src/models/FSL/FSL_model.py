# from jmlib.data import Data
# from jmlib.data.loaders import PTBXL

# from jmlib.data.processing.common import LambdaModule
# from jmlib.data.writers.common import Writer
# from jmlib.data.generators.fewshot import FewShotGenerator
# from jmlib.data.splitters.common import ClassSplitter 

from typing import Unpack
from keras.layers import TimeDistributed, Lambda, Activation, Flatten, Dense
from keras.layers import BatchNormalization, MaxPooling1D, Conv1D
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy

from FSL.common import BaseModel, BaseModelParams
from FSL.utils import reduce_tensor, reshape_query, proto_dist, LinearFusion
import numpy as np
import matplotlib.pyplot as plt

# ptbxl = Data(name="raw_PTBXL", verbose=True)
# ptbxl.add(
#     PTBXL(data_dir="/Users/jbthompson/Documents/final_folder/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
#     ClassSplitter({"train": 0.6, "val": 0.2, "test": 0.2}),
#     FewShotGenerator(way=5, shot=5, query= 5, batch_size=100)
# )
# ptbxl.run()


KERNEL_TUPLE = tuple[tuple, tuple]
EPOCHS = 100

class PMCNN(BaseModel):
    """Zicong Li et al., 2022 - https://doi.org/10.1109/BHI56158.2022.9926948.

    Implementation by Joe McMahon. Adapted from from code provided by
    Zicong Li. Summary from paper: "a parallel multi-scale CNN (PM-CNN) based
    prototypical network for arrhythmia classification".
    This implementation supports categorical classification.
    Designed for use on the CPSC-2018 dataset.

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate.
    depth : int, default=4
        Number of convolutional layers in the prototypical network.
    fd : int, default=512
        The shape of the 1D feature vector output of the fusion layer.
    kernels : tuple of tuples of ints, default=((3,3),(7,7))
        Convolutional kernel shapes.
    filters : int, default=64
        Number of filters in convolutional layers.
    **kwargs
        keyword arguments to pass to super class. See jmlib.models.BaseClass.
    """

    LR: float
    DEPTH: int
    FD: int
    KERNELS: KERNEL_TUPLE
    FILTERS: int

    def __init__(self,
                 lr: float = 0.001,
                 depth: int = 4,
                 fd: int = 512,
                 kernels: KERNEL_TUPLE = (3, 7),
                 filters: int = 64,
                 **kwargs: Unpack[BaseModelParams]):#check on other github??
        self.LR = lr
        self.DEPTH = depth
        self.FD = fd
        self.KERNELS = kernels
        self.FILTERS = filters
        super().__init__(**kwargs)

    def _layers(self, X):
        Xs, Xq = X
        shot = Xs.shape[-4]
        query = Xq.shape[-4]

        proto_model3 = TimeDistributed(
            self._proto_model(self.KERNELS[0]), name="Prototype_CNN_3"
        )
        proto_model7 = TimeDistributed(
            self._proto_model(self.KERNELS[1]), name="Prototype_CNN_7"
        )

        Xs3 = proto_model3(Xs)
        Xq3 = proto_model3(Xq)

        Xs7 = proto_model7(Xs)
        Xq7 = proto_model7(Xq)

        Xs = LinearFusion(shot, self.FD)(Xs7, Xs3)
        Xq = LinearFusion(query, self.FD)(Xq7, Xq3)

        Xs = Lambda(reduce_tensor, name="Reduce_Support")(Xs)
        Xq = Lambda(reshape_query, name="Reshape_Query")(Xq)

        X = Lambda(proto_dist, name="Prototype_Distance")([Xs, Xq])

        return X

    def _proto_model(self, k) -> Sequential:
        cnn = Sequential()
        for _ in range(self.DEPTH):  # type: ignore
            cnn.add(Conv1D(self.FILTERS, k, padding='same'))
            cnn.add(BatchNormalization())
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D())

        cnn.add(Flatten())
        cnn.add(Dense(self.FD))
        return cnn

    @property
    def optimizer(self):
        """Adam Optimizer."""
        return Adam(learning_rate=self.LR)  # type: ignore

    @property
    def loss(self):
        """Categorical Crossentropy Loss."""
        return CategoricalCrossentropy()

    @property
    def callbacks(self):
        """Callbacks."""
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=2,
            min_lr=1e-8,  # type: ignore
            cooldown=2
        )
        return [reduce_lr]
    
