import numpy as np
from numpy.typing import NDArray
from scipy import signal, linalg, fft



def __range_fft(data: NDArray, rwin: NDArray = None, n: int = None) -> NDArray:
    shape = np.shape(data)

    if rwin is None:
        rwin = 1
    else:
        rwin = np.tile(rwin[np.newaxis, np.newaxis, ...], (shape[0], shape[1], 1))

    return fft.fft(data * rwin, n=n, axis=2)

def __doppler_fft(data: NDArray, dwin: NDArray = None, n: int = None) -> NDArray:
    shape = np.shape(data)

    if dwin is None:
        dwin = 1
    else:
        dwin = np.tile(dwin[np.newaxis, ..., np.newaxis], (shape[0], 1, shape[2]))

    return fft.fft(data * dwin, n=n, axis=1)



def range_doppler_fft(baseband: NDArray) -> NDArray:
    "Reduce to 2D array (sum over angle), compute FFTs (doppler, range), returns magnitude of complex numbers (np.abs)"
    
    # convert to np array
    baseband = np.array(baseband, dtype=np.complex64)
    
    if (len(baseband.shape) > 2):
        baseband = np.sum(baseband, axis=0, keepdims=True) # sum all RX channels -> RX beamforming at zero deg

    _, chirps, range_samples = baseband.shape

    # get windows
    range_window = signal.windows.chebwin(range_samples, at=90) 
    doppler_window = signal.windows.chebwin(chirps, at=60)

    rn = 256 # range point_FFT 
    dn = 1024 # doppler point_FFT

    # do FFTs
    doppler_range = __doppler_fft(__range_fft(baseband, rwin=range_window, n=rn), dwin=doppler_window, n=dn)

    return np.abs(doppler_range[0])