import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import scipy.signal
from typing import Union
from us.utils.types import Real


class Image:
    # Constructor
    def __init__(self,
                 limits,
                 data: np.ndarray):

        self.__limits = limits
        self.__data = data

    # Properties
    @property
    def image_limits(self):
        return self.__limits

    @property
    def data(self):
        return self.__data

    # Methods
    def bmode(self, normalize: Union[str, float] = None) -> np.ndarray:
        env = self.envelope(normalize=normalize)
        return np.array([20]) * np.log10(env)

    def envelope(self, normalize: Union[str, float] = None) -> np.ndarray:
        env = np.abs(scipy.signal.hilbert(self.data, axis=-1))
        if normalize is not None:
            if normalize == 'max':
                env /= env.max()
            elif isinstance(normalize, float):
                env /= normalize
            else:
                raise NotImplemented('Unsupported normalization type')
        return env

    def plot_bmode(self, normalize: Union[str, float] = 'max', db_range: int = 60, axis_scale: Real = 1e3,
                   cbar: bool = False, show_axes: bool = True, ax=None, save_path: str = None) -> None:
        #
        if save_path is not None:
            matplotlib.rcParams.update({'font.size': 14})

        if ax is None:
            plt.figure()
            ax = plt.gca()

        extent = None
        x_lim, z_lim = self.image_limits
        if x_lim is not None and z_lim is not None:
            extent = [x_lim[0], x_lim[1], z_lim[1], z_lim[0]]
            extent = [val * axis_scale for val in extent]

        bmode = self.bmode(normalize=normalize)

        im = ax.imshow(bmode.T, cmap='gray', vmin=-db_range, vmax=0, extent=extent, interpolation=None)

        if not show_axes:
            ax.set_axis_off()

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')

        if cbar:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            plt.colorbar(im, cax=cax, label='dB [─]')

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
