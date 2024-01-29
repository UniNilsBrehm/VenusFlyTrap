import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from IPython import embed
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def butter_filter_design(filter_type, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.9999
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    return b, a


def butter_filter(data, filter_type, cutoff, fs, order=2):
    b, a = butter_filter_design(filter_type, cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def elu(x, x0, a, tau):
    if x <= x0:
        return 0
    else:
        return a * (1-np.exp(-(x-x0)/tau))


def elu_fit(t, t0, a, tau):
    return list(map(lambda x: elu(x, t0, a, tau), t))


def double_exp(x, a, b, c, d):
    return a*(1-np.exp(-b*x)) + c*(1-np.exp(-d*x))


def exp_fit(x, a, b):
    return a*(1-np.exp(-b*x))


def exp_growth(x, L, b, k):
    return L - (L - b)*np.exp(-k*x)


rec_date = '20231026'
rec_name = 'data_01'
base_dir = Path('C:/UniFreiburg/VenusFlyTrap/recordings/') / rec_date / rec_name
e_file_dir = base_dir / f'{rec_date}_ephys_{rec_name}.csv'
force_file_dir = base_dir / f'{rec_date}_force_{rec_name}.csv'
df_force = pd.read_csv(force_file_dir.as_posix())
fr = 1000
fr_force = 10

force_time = df_force["Time"]
force_values = df_force["Force"].to_numpy()

cut_out = force_values[992:1200]
# cut_out = force_values[7359:7478]
cut_out = cut_out - np.min(cut_out)
xdata = np.arange(0, cut_out.shape[0], 1) / 10

# cut_out = butter_filter(data=cut_out, filter_type='low', cutoff=0.5, fs=fr_force, order=2)


# p0 = [0, 0.5, 10]
# popt, pcov = curve_fit(exp_fit, xdata, cut_out, method='lm', maxfev=5000)
popt, pcov = curve_fit(exp_growth, xdata, cut_out, method='lm', maxfev=5000)

# y_fit = exp_fit(xdata, popt[0], popt[1])

# L(upper bound), b(lower bound), k(growth rate)
y_fit = exp_growth(xdata, popt[0], popt[1], popt[2])

plt.plot(xdata, cut_out, 'k')
plt.plot(xdata, y_fit, 'r')
plt.show()

embed()
exit()