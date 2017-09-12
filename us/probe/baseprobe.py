import numpy as np
from typing import Optional, Union
from abc import ABCMeta, abstractmethod

# Types
Real = Union[int, float]


class BaseProbe(metaclass=ABCMeta):
    def __init__(self,
                 name: str,
                 center_frequency: Real,
                 element_number: int,
                 element_width: Real,
                 element_height: Optional[Real] = None,
                 bandwidth: Optional[Real] = None):

        self.__name = name
        self.__center_frequency = center_frequency
        self.__element_number = element_number
        self.__element_width = element_width
        self.__element_height = element_height
        self.__bandwidth = bandwidth

    # Properties
    @property
    def center_frequency(self):
        return self.__center_frequency

    @property
    def element_number(self):
        return self.__element_number

    @property
    def element_width(self):
        return self.__element_width

    @property
    def element_height(self):
        return self.__element_height

    @property
    def bandwidth(self):
        return self.__bandwidth

    @property
    @abstractmethod
    def element_positions(self): pass

    @property
    @abstractmethod
    def width(self) -> Real: pass

    @property
    @abstractmethod
    def height(self) -> Real: pass

    # methods
    @abstractmethod
    def impulse_response(self, sampling_frequency) -> np.ndarray: pass
