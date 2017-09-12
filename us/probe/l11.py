from .probe1d import Probe1D


class L11Probe(Probe1D):
    def __init__(self,
                 name='L11-4v',
                 pitch=300e-6,
                 center_frequency=1540 / 300e-6,
                 element_number=128,
                 element_width=0.27e-3,
                 element_height=None,
                 bandwidth=0.67,
                 impulse_cycles=2,
                 impulse_window='gauss'):

        super(L11Probe, self).__init__(name=name,
                                       pitch=pitch,
                                       center_frequency=center_frequency,
                                       element_number=element_number,
                                       element_width=element_width,
                                       element_height=element_height,
                                       bandwidth=bandwidth,
                                       impulse_cycles=impulse_cycles,
                                       impulse_window=impulse_window)
