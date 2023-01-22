import numpy as np
from scipy.signal import hilbert, cheby1, filtfilt

def interval(positions):
    intervals = []
    n = positions.shape[0]
    intervals = np.zeros((n-1,1))
    for i in range(n-1):
        intervals[i] = positions[i+1] - positions[i]
    even_intervals = intervals[0::2] 
    odd_intervals = intervals[1::2]  
    return intervals, even_intervals, odd_intervals

def cheby1_bandpass_filter(data, lowcut, highcut, fs, order=5, rp=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, rp=rp, Wn=[low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def clear_adjacents(peaks, interval):
    indices = []
    for i in range(len(peaks) - 1):
        diff = peaks[i+1] - peaks[i]
        if diff < 0.25 * interval:
            indices.append(i+1)
    return np.delete(peaks,indices)
        
        
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]    