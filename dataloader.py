#I think we should include something that changes the value after the furier transform when deviding by 0 (this happens for train_7 it also gives a warning but the code still runs) so that the value is not set to -inf. Or maybe we can fix this already as a preprocessing step. I am just unsure how exactly we want to handle this!!
import numpy as np
from visualise import plot_digi_fig


# FFT conversion function code goes here
def FFT_transform(mat):
    #tranform img data to FFT
    fft_data = np.fft.fft2(mat)
    #tranform complex number matrix to floating point
    fshift = np.fft.fftshift(fft_data)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    return magnitude_spectrum

# Code of functions to load train/test data from file goes here
with open('Data/train_9.txt') as f:
    for line in f:
        curr = line.strip()
        mat = np.fromstring(curr, dtype=int, sep='  ')
        mat_r = np.reshape(mat, (-1, 15))

        mag_val = FFT_transform(mat_r)
        plot_digi_fig(mag_val)
        break