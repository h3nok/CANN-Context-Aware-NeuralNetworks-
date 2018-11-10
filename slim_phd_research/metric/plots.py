import pandas as pd
from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz


df = pd.read_csv("C:\phd\Analysis\InceptionV4\Total loss\\all.csv")
print (df.head(5))
data = {'4x4':df['4x4'],'8x8': df['8x8'],'16x16': df['16x16'], 'CIFAR100':df['CIFAR100']}
color_map = {'4x4':'darkblue', '8x8':'m','16x16':'red','CIFAR100':'c'}
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

for label, value in data.items():
    print(label)

    # Filter requirements.
    order = 6
    fs = 16.0       # sample rate, Hz
    cutoff = 0.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 4.0         # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data_y = value
    data_x = df['Step']/1000
    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data_y, cutoff, fs, order)
    x = butter_lowpass_filter(data_x, cutoff,fs,order)
    plt.subplot()
        # plt.plot(t, data, 'b-', label='data')
    plt.plot(x, y, 'g-', linewidth=2, label=label,color=color_map[label])
    plt.xlabel(' Training steps[Thousands]')
    plt.ylabel(' Loss')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
plt.show()