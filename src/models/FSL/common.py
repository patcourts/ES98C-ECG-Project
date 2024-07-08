from typing import Optional, TypedDict
from abc import ABC, abstractmethod

from numpy.random import Generator as NPGenerator, default_rng
from keras import Model as KerasModel


class BaseModelParams(TypedDict, total=False):
    """TypedDict for type-hinting kwargs passed to BaseClass from children."""

    input_tensors: tuple | list
    seed: Optional[int]
    name: Optional[str]
    verbose: bool


class BaseModel(ABC):
    """Base Class for writing Keras models.

    A simple wrapper inheriting the keras.models.Model class which provides
    access to an ModelOptions instance.

    Parameters
    ----------
    input_tensors : tensor
        Input tensor to pass to model.
    seed : int, optional
    name : str, optional
    verbose : bool, default=False

    Attributes
    ----------
    NAME : str
        Name of model for use in logging/outputs.
    SEED : int, optional
        Seed for generating randomness deterministically.
    rng : np.random.Generator
        np.random.Generator instance used throughout the model.
    vb : bool, default=False
        verbosity, True = verbose, False = silent.
    optimizer
    loss
    model
    """

    NAME: str
    SEED: Optional[int] = None
    rng: NPGenerator
    vb: bool

    def __init__(self,
                 input_tensors,
                 seed: Optional[int] = None,
                 name: Optional[str] = None,
                 verbose: bool = False):
        self.NAME = name or self.__class__.__qualname__
        self.SEED = seed
        self.rng = default_rng(seed)
        self.vb = verbose

        X = input_tensors
        Y = self._layers(X)
        self.__model = KerasModel(X, Y)

    @abstractmethod
    def _layers(self, X):
        """Abstract function defining model layers.

        This function should be implemented to define layers using the Keras
        functional API to be passed to keras.Model constructor.

        Parameters
        ----------
        X : tensor
            Keras input tensor.

        Returns
        -------
        tensor
            Keras output tensor.

        Examples
        --------
        class MyModel(BaseModel):

            ...

        def _layers(self, X):
            X = keras.layers.RandomCrop(width=128, height=128)(X)
            X = keras.layers.Conv2D(filters=32, kernel_size=3)(X)
            X = keras.layers.GlobalAveragePooling2D()(X)
            X = keras.layers.Dense(10)(X)
            return X
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def optimizer(self):
        """Implement optimizer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def loss(self):
        """Implement loss function."""
        raise NotImplementedError

    @property
    def model(self) -> KerasModel:
        """Internal keras model."""
        return self.__model