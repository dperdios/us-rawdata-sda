from typing import Optional

import numpy as np
import scipy.signal

from .baseprobe import BaseProbe
from us.utils.types import Real


class Probe1D(BaseProbe):
    def __init__(self,
                 name: str,
                 pitch: Real,
                 center_frequency: Real,
                 element_number: int,
                 element_width: Optional[Real] = None,
                 element_height: Optional[Real] = None,
                 bandwidth: Optional[Real] = None,
                 impulse_cycles: Optional[Real] = None,
                 impulse_window: Optional[str] = None):

        if element_width is None:
            element_width = pitch

        super(Probe1D, self).__init__(name=name,
                                      center_frequency=center_frequency,
                                      element_number=element_number,
                                      element_width=element_width,
                                      element_height=element_height,
                                      bandwidth=bandwidth)

        self.__type = '1D'
        self.__pitch = pitch
        self.__impulse_cycles = impulse_cycles
        self.__impulse_window = impulse_window

    # Properties
    @property
    def pitch(self):
        return self.__pitch

    @property
    def impulse_cycles(self):
        return self.__impulse_cycles

    @property
    def impulse_window(self):
        return self.__impulse_window

    #   Computed properties
    @property
    def width(self):
        return (self.element_number - 1) * self.pitch

    @property
    def height(self):
        return None

    @property
    def element_positions(self):
        return np.linspace(start=-self.width / 2, stop=self.width / 2, num=self.element_number)

    # Methods
    def impulse_response(self, sampling_frequency):
        dt = 1 / sampling_frequency
        if self.impulse_window in ['hanning', 'blackman']:
            t_start = 0
            t_stop = int(self.impulse_cycles / self.center_frequency * sampling_frequency) * dt  # int() applies floor
            t_num = int(self.impulse_cycles / self.center_frequency * sampling_frequency) + 1  # int() applies floor
            t = np.linspace(t_start, t_stop, t_num)
            impulse = np.sin(2 * np.pi * self.center_frequency * t)
            if self.impulse_window == 'hanning':
                win = np.hanning(impulse.shape[0])
            elif self.impulse_window == 'blackman':
                win = np.blackman(impulse.shape[0])
            else:
                raise NotImplementedError('Window type {} not implemented'.format(self.impulse_window))
            return impulse * win
        elif self.impulse_window == 'gauss':
            # Compute cutoff time for when the pulse amplitude falls below `tpr` (in dB) which is set at -100dB
            tpr = -60
            t_cutoff = scipy.signal.gausspulse('cutoff', fc=int(self.center_frequency), bw=self.bandwidth, tpr=tpr)
            t = np.arange(-t_cutoff, t_cutoff, dt)
            return scipy.signal.gausspulse(t, fc=self.center_frequency, bw=self.bandwidth, tpr=tpr)
        else:
            raise NotImplementedError('Window type {} not implemented'.format(self.impulse_window))
