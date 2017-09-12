import numpy as np
import scipy.interpolate

from .baseseq import BaseSequence
from us.image import Image
from us.probe import BaseProbe, Probe1D
from us.utils.types import Real, Data, Vector


class PWSequence(BaseSequence):
    def __init__(self,
                 name: str,
                 probe: BaseProbe,
                 sampling_frequency: Real,
                 transmit_frequency: Real,
                 transmit_wave: str,
                 transmit_cycles: Real,
                 mean_sound_speed: Real,
                 medium_attenuation: Real,
                 angles: Vector,
                 initial_times: Vector,
                 data: Data):

        sequence_type = 'PW'

        super(PWSequence, self).__init__(name=name,
                                         sequence_type=sequence_type,
                                         probe=probe,
                                         sampling_frequency=sampling_frequency,
                                         transmit_frequency=transmit_frequency,
                                         transmit_wave=transmit_wave,
                                         transmit_cycles=transmit_cycles,
                                         mean_sound_speed=mean_sound_speed,
                                         medium_attenuation=medium_attenuation,
                                         initial_times=initial_times,
                                         data=data)

        self.__angles = angles

    # Properties
    @property
    def angles(self):
        return self.__angles

    @property
    def image_limits(self):
        x_min = -self.probe.width / 2
        x_max = self.probe.width / 2

        # Image limits are computed assuming a normal incidence PW (even if zero angle is not available)
        ind_zero_angle_check = np.where(self.angles == 0)[0]
        if ind_zero_angle_check.size == 0:
            ind_angle = 0
        else:
            ind_angle = int(ind_zero_angle_check)  # lists require int or slice for indexing

        initial_time = self.initial_times[ind_angle]

        sample_number = self.sample_numbers[ind_angle]

        z_min = initial_time * self.mean_sound_speed / 2
        z_max = (initial_time + sample_number * 1 / self.sampling_frequency) * self.mean_sound_speed / 2

        return [[x_min, x_max], [z_min, z_max]]

    # Methods
    def beamform(self, dx_ratio: float = 1/3, dz_ratio: float = 1/4, interp: str ='cubic') -> Image:
        ########################################################
        # Very basic delay-and-sum (DAS) beamformer
        # ONLY VALID FOR ZERO ANGLE PLANE WAVE in 2D FOR NOW
        ########################################################
        error_str = 'Only normal incidience plane wave implemented in 2D'
        probe = self.probe
        if len(self.angles) > 1 or len(self.initial_times) > 1 or len(self.data) > 1:
            raise NotImplemented(error_str)
        if not self.angles[0] == 0:
            raise NotImplemented(error_str)
        if not isinstance(probe, Probe1D):
            raise NotImplemented(error_str)
        if interp == 'cubic':
            k = 3
        elif interp == 'linear':
            k = 1
        else:
            raise NotImplemented('Unsupported interpolation method {}'.format(interp))

        sample_number = self.sample_numbers[0]
        data = self.data[0]
        dtype = data.dtype

        time_samples = self.initial_times[0] + (np.arange(sample_number) + 1) * 1 / self.sampling_frequency
        time_samples = time_samples.astype(dtype=dtype)

        lat_pitch_ratio = probe.pitch / (dx_ratio * self.wavelength)

        x_im_lim, z_im_lim = self.image_limits
        lateral_image_positions = np.linspace(start=x_im_lim[0], stop=x_im_lim[1],
                                              num=int(np.ceil(lat_pitch_ratio * probe.element_number)),
                                              dtype=dtype)

        t_offset = 1 / self.sampling_frequency
        t_im_min = 2 * z_im_lim[0] / self.mean_sound_speed + t_offset
        t_im_max = 2 * z_im_lim[1] / self.mean_sound_speed + t_offset

        ax_sample_ratio = self.mean_sound_speed / self.wavelength / dz_ratio / self.sampling_frequency
        image_axial_time = np.linspace(start=t_im_min, stop=t_im_max,
                                       num=int(np.ceil(ax_sample_ratio * sample_number)),
                                       dtype=dtype)

        # Pre-computations
        T, P = np.meshgrid(image_axial_time, probe.element_positions)
        T = T.astype(dtype=dtype)
        P = P.astype(dtype=dtype)
        beamformed_data = np.zeros([np.size(lateral_image_positions, axis=0), np.size(image_axial_time, axis=0)],
                                   dtype=dtype)
        tx_delay = 0.5 * T

        for it_x, pos in enumerate(lateral_image_positions):
            # Pre-computation
            t_pre = (pos - P) / self.mean_sound_speed

            # Round trip time of flight
            rx_delay = np.sqrt(t_pre**2 + tx_delay**2)
            txrx_delay = tx_delay + rx_delay

            # Obliquity factor according to Selfridge et al. - A theory for the radiation pattern of a narrow-strip
            # acoustic transducer, Applied Physics Letters, 1980
            obliquity = tx_delay / rx_delay * np.sinc(probe.element_width / self.wavelength * t_pre / rx_delay)

            for obl, t, d in zip(obliquity, txrx_delay, data):
                f = scipy.interpolate.InterpolatedUnivariateSpline(time_samples, d, k=k, ext='zeros')
                beamformed_data[it_x] += obl * f(t)

        image = Image(self.image_limits, beamformed_data)

        return image
