from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple
import numpy as np

from us.image import Image
from us.probe import BaseProbe
from us.utils.types import Real, Data, Vector


class BaseSequence(metaclass=ABCMeta):
    def __init__(self,
                 name: str,
                 sequence_type: str,
                 probe: BaseProbe,
                 sampling_frequency: Real,
                 transmit_frequency: Real,
                 transmit_wave: str,
                 transmit_cycles: Real,
                 mean_sound_speed: Real,
                 medium_attenuation: Real,
                 initial_times: Vector,
                 data: Data):

        self.__name = name
        self.__type = sequence_type
        if isinstance(probe, BaseProbe):
            self.__probe = probe
        else:
            raise TypeError('Argument `probe` must be an instance of class {name}'.format(name=BaseProbe.__name__))
        self.__sampling_frequency = sampling_frequency
        self.__transmit_frequency = transmit_frequency
        self.__transmit_wave = transmit_wave
        self.__transmit_cycles = transmit_cycles
        self.__mean_sound_speed = mean_sound_speed
        self.__medium_attenuation = medium_attenuation
        self.__initial_times = initial_times
        self.__data = data

    # Properties
    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def probe(self):
        return self.__probe

    @property
    def sampling_frequency(self):
        return self.__sampling_frequency

    @property
    def transmit_frequency(self):
        return self.__transmit_frequency

    @property
    def transmit_wave(self):
        return self.__transmit_wave

    @property
    def transmit_cycles(self):
        return self.__transmit_cycles

    @property
    def transmit_cycles(self):
        return self.__transmit_cycles

    @property
    def mean_sound_speed(self):
        return self.__mean_sound_speed

    @property
    def medium_attenuation(self):
        return self.__medium_attenuation

    @property
    def initial_times(self):
        return self.__initial_times

    @property
    def data(self):
        return self.__data

    # Additional properties
    @property
    def wavelength(self):
        return self.mean_sound_speed / self.probe.center_frequency

    @property
    @abstractmethod
    def image_limits(self): pass

    @property
    def sample_numbers(self):
        return [d.shape[1] for d in self.data]

    # Methods
    @abstractmethod
    def beamform(self) -> Image: pass

    def normalize_data(self) -> None:
        for data in self.data:
            data /= np.abs(data).max()

    def update_data(self, data_new: Data) -> None:
        # Check that data_new has the same dimensions
        old_size = [d.shape for d in self.data]
        new_size = [d.shape for d in data_new]
        if len(old_size) == len(new_size) and all([o == n for o, n in zip(old_size, new_size)]):
            self.__data = data_new
        else:
            raise TypeError('The `data_new` must have the exact same length and shape')

    def _tgc_exp_factor(self) -> Vector:
        # Convert `medium_attenuation` to SI units, i.e. in Np/Hz/m
        db_to_neper = 1 / (20 * np.log10(np.exp(1)))
        m_to_cm = 1e-2
        hz_to_mhz = 1e6
        alpha_si = self.medium_attenuation * db_to_neper / hz_to_mhz / m_to_cm

        gain = []
        for initial_time, data in zip(self.initial_times, self.data):
            sample_number = data.shape[-1]
            depth = initial_time * self.mean_sound_speed / 2 + np.arange(
                sample_number) * self.mean_sound_speed / (2 * self.sampling_frequency)
            gain.append(np.exp(2 * alpha_si * self.probe.center_frequency * depth))  # roundtrip

        return gain

    def apply_tgc(self) -> None:
        tgc_factors = self._tgc_exp_factor()
        for tgc_fact, data in zip(tgc_factors, self.data):
            data *= tgc_fact

    def crop_data(self,
                  first_index: Union[Real, List[Real], Tuple[Real, ...], np.ndarray],
                  last_index: Union[Real, List[Real], Tuple[Real, ...], np.ndarray]) -> None:
        array_types = (list, tuple, np.ndarray)
        if not isinstance(first_index, (int, *array_types)):
            raise TypeError('first_index must be an int or an array-like')
        if not isinstance(last_index, (int, list, tuple, np.ndarray)):
            raise TypeError('last_index must be an int or an array-like')
        if isinstance(first_index, int):
            first_indexes = [first_index for _ in range(len(self.data))]
        elif isinstance(first_index, array_types):
            first_indexes = first_index
        else:
            raise TypeError
        if isinstance(last_index, int):
            last_indexes = [last_index for _ in range(len(self.data))]
        elif isinstance(last_index, array_types):
            last_indexes = last_index
        else:
            raise TypeError

        # Loop over data and initial times
        data_crop = []
        initial_times_update = []
        for data, init_time, f_ind, l_ind in zip(self.data, self.initial_times, first_indexes, last_indexes):
            data_crop.append(data[:, first_index:last_index])
            initial_times_update.append(init_time + first_index * 1 / self.sampling_frequency)

        # Set data and corresponding initial_times
        self.__data = data_crop
        self.__initial_times = initial_times_update
