import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from IPython import embed
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def mixed_exp(x, a, b, tau1):
    aa = np.zeros_like(x)
    aa[x >= a] = 1

    # bb = np.zeros_like(x)
    # bb[x > b] = x[x > b]
    #
    # cc = np.zeros_like(x)
    # cc[x > c] = x[x > c]
    
    # return aa * (b*(1 - np.exp(-(x-a)/tau1)) + c*(1-np.exp(-((x-a)/tau2))))
    return aa * (b*(1 - np.exp(-(x-a)/tau1)))
    # return aa * (b*(1 - (np.exp(-(x-a)/tau1) * np.exp(-(x-a)/tau2))))


def sigmoid(x, L, x0, k, b):
    # Logistic Function
    # y = L / (1 + np.exp(-k*(x-x0))) + b
    return L / (1 + np.exp(-k*(x-x0))) + b


def moving_average(data, win):
    filtered_data = np.convolve(data, np.ones(win) / win, mode='same')
    return filtered_data


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


def notch_filter(data, f0, q_factor, fs):
    b, a = signal.iirnotch(f0, q_factor, fs)
    y = signal.filtfilt(b, a, data)
    return y


# SETTINGS
rec_date = '20231026'
rec_name = 'data_01'
base_dir = Path('C:/UniFreiburg/VenusFlyTrap/recordings/') / rec_date / rec_name
e_file_dir = base_dir / f'{rec_date}_ephys_{rec_name}.csv'
force_file_dir = base_dir / f'{rec_date}_force_{rec_name}.csv'
df_e = pd.read_csv(e_file_dir.as_posix(), index_col=0)
df_force = pd.read_csv(force_file_dir.as_posix())
fr = 1000
fr_force = 10

y_data = df_e['Voltage']
t_data = df_e['Time']
stimulus = df_e['Stimulus']
force_time = df_force["Time"]
force_values = df_force["Force"]


# Get Time Alignment in secs (if already there)
try:
    time_align = pd.read_csv(base_dir / f'{rec_date}_alignment_{rec_name}.txt')
    time_diff = (time_align['force_end'] - time_align['e_end']).item()
    force_time = force_time - time_diff
except FileNotFoundError:
    time_diff = 0
    print('Could not find alignment file')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Remove Artifacts manually
if rec_name == 'data_06':
    start_time = 60
    end_time = 70
    y_data[int(start_time*fr):int(end_time*fr)] = y_data[int(start_time*fr)]
    print('Removed Artifacts')

if rec_name == 'data_11':
    start_time = 285
    end_time = 292
    y_data[int(start_time*fr):int(end_time*fr)] = y_data[int(start_time*fr)]
    print('Removed Artifacts')

# Filter Voltage Trace
# Notch Filter (Remove 50 Hz Noise)
y_data_notch = notch_filter(data=y_data, f0=50, q_factor=30, fs=fr)

# Low Pass Filter (Remove High Frequency Components)
cut_off = 100
y_data_lp = butter_filter(data=y_data_notch, filter_type='low', cutoff=cut_off, fs=fr, order=2)

# High Pass Filter (Remove DC Component)
# cut_off = 0.001
cut_off = 0.01
y_data_fil = butter_filter(data=y_data_lp, filter_type='high', cutoff=cut_off, fs=fr, order=2)

# Filter Force Trace
force_values_fil = butter_filter(data=force_values, filter_type='low', cutoff=1, fs=fr_force, order=2)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Get settings for AP peak detection
try:
    detection_settings = pd.read_csv(base_dir / f'{rec_date}_AP_detection_{rec_name}.txt', header=0)
    peak_height_th = detection_settings['peak_height'].item()
    peak_distance_time = detection_settings['peak_distance'].item()
    peak_min_width = detection_settings['min_width'].item()
    peak_max_width = detection_settings['max_width'].item()
except FileNotFoundError:
    print('==== Could not find AP DETECTION SETTINGS! Will use default values ====')
    peak_height_th = 20  # in mV
    peak_distance_time = 2  # in secs
    peak_min_width = 0.1  # in secs
    peak_max_width = 2  # in secs

print('')
print('================================================')
print('AP DETECTION SETTINGS:')
print(f'height threshold: {peak_height_th} mV')
print(f'distance threshold: {peak_distance_time} s')
print(f'min width: {peak_min_width} s')
print(f'max width: {peak_max_width} s')
print('================================================')
print('')

# Now find peaks in voltage signal
peaks, _ = find_peaks(
    y_data_fil*-1,  # invert signal
    height=peak_height_th,
    distance=int(peak_distance_time * fr),
    width=(int(peak_min_width * fr), int(peak_max_width * fr))
)

if rec_name == 'data_07':
    peaks[1] = peaks[1] - 600
    print('AP Detection has been manually corrected!')

# Find Stimulus times
th = 1
stimulus_diff = np.diff(stimulus, append=0)
peaks_stim, _ = find_peaks(stimulus_diff, height=th)

force_diff = np.diff(force_values_fil, append=0)

try:
    aps_numbers = list(pd.read_csv(base_dir / f'{rec_date}_AP_numbers_{rec_name}.txt', header=None).to_numpy()[0].astype('int'))
    # t0_vals = list(pd.read_csv(base_dir / f'{rec_date}_AP_numbers_{rec_name}.txt', header=None).to_numpy()[1])
    aps_bool = True
except FileNotFoundError:
    print('Could not find selected APs!')
    aps_bool = False


# PLOT
fig, axs = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(18, 10, forward=True)
fig.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.05, hspace=0.05)
axs[0].plot(t_data, y_data_fil, 'k')
axs[0].plot((peaks / fr), y_data_fil[peaks], 'ro')
axs[0].set_ylabel('Surface Potential [mV]')

axs[1].plot(force_time, force_values, 'k', lw=2)
# axs[1].plot(force_time, force_values_fil, 'g', lw=2)
# axs[1].plot(force_time, np.diff(force_values_fil, append=0), 'y', lw=2)
axs[1].plot([(peaks_stim / fr), (peaks_stim / fr)], [np.zeros_like(peaks_stim), np.zeros_like(peaks_stim) + 0.5],
            'k:', alpha=0.1)
axs[1].plot([(peaks / fr), (peaks / fr)], [np.zeros_like(peaks), np.zeros_like(peaks) + 0.5],
            'r-', alpha=0.1)

axs[1].set_ylabel('Force [N]')

axs[2].plot(t_data, stimulus, 'k')
axs[2].plot((peaks_stim / fr), np.zeros_like(peaks_stim) + 3, 'ro')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Speaker Voltage')

# fig.savefig(base_dir / f'{rec_date}_{rec_name}_overview.png', dpi=300)
# plt.show()

if aps_bool:
    results = []
    axs[1].plot([(peaks[aps_numbers] / fr), (peaks[aps_numbers] / fr)], [np.zeros_like(peaks[aps_numbers]), np.zeros_like(peaks[aps_numbers] ) + 0.5],
                'r-', alpha=0.8)
    # in secs
    time_before_ap = 1
    cut_dur = time_before_ap + 30

    # For data_07
    # cut_dur = time_before_ap + 40

    # t0_vals = [23.348, 99, 735.72, 1067, 1521]
    print('')
    cc = 0
    for ap in aps_numbers:
        ap_onset_time = peaks[ap] / fr
        start_time = ap_onset_time - time_before_ap
        # ap_idx_force = int(ap_time * fr_force)
        ap_idx_force = int(start_time * fr_force)
        force_cutout = force_values_fil[ap_idx_force:ap_idx_force+int(cut_dur*fr_force)]

        # Correct for time alignment (time_diff)!
        xdata = (np.arange(ap_idx_force, ap_idx_force+force_cutout.shape[0], 1) / fr_force) - time_diff

        p0 = [max(force_cutout), np.median(xdata), 1, min(force_cutout)]  # this is an mandatory initial guess

        try:
            # popt, pcov = curve_fit(sigmoid, xdata, force_cutout, p0, method='lm')
            popt, pcov = curve_fit(mixed_exp, xdata-np.min(xdata), force_cutout-np.min(force_cutout), method='lm')

        except RuntimeError:
            print(f'AP {ap}: Got "RuntimeError", what is wrong?')
            embed()
            exit()

        # sigmoid(x, L ,x0, k, b):
        # y_fit = sigmoid(xdata, popt[0], popt[1], popt[2], popt[3])

        # mixed_exp(x, a, b, tau1)
        y_fit = mixed_exp(xdata-np.min(xdata), popt[0], popt[1], popt[2]) + np.min(force_cutout)

        # plt.plot(xdata-np.min(xdata), force_cutout, 'k')
        # plt.plot(xdata-np.min(xdata), y_fit, 'r')
        # plt.show()

        # Error Estimate of Parameters
        perr = np.sqrt(np.diag(pcov))

        # Exponential Fit
        # Time from AP to Force Increase
        force_onset = popt[0] - time_before_ap - time_diff
        force_time_constant = popt[2]
        force_time_constant_err = perr[2]
        force_min = np.min(y_fit)
        # force_max = popt[1] + np.min(force_cutout)
        force_max = np.max(y_fit)
        # force_max_err = perr[1]

        print(f'AP: {ap} at {ap_onset_time:.3f} secs:')
        print(f'Force increases from: {force_min:.4f} to {force_max:.4f}, '
              f'delta = {force_max-force_min:.4f}, '
              f'tau = {force_time_constant:.4f} ({force_time_constant_err:.4f}), '
              f'latency = {force_onset:.4f} s')
        print('')

        # Collect Results
        # ['Date', 'RecName', 'AP_Nr', 'minForce', 'maxForce', 'deltaForce', 'APtime', 'TimeConstant', 'Latency', 'ID']
        results.append(
            [rec_date,
             rec_name,
             ap,
             force_min,
             force_max,
             force_max-force_min,
             ap_onset_time,
             force_time_constant,
             force_onset,
             cc
             ]
        )
        cc += 1

        # SIGMOID FIT
        # # Min. and Max. Force
        # force_min = popt[1] + np.min(force_cutout)
        # force_min_err = perr[1]
        # force_max = popt[0] + np.min(force_cutout)
        # force_max_err = perr[0]
        #
        # # Steepness
        # steepness = popt[2]
        # steepness_err = perr[2]
        # # X0 (Time of Half Width)
        # x0 = popt[1]
        #
        # # Latency
        # # Difference between AP time and Half Width (xo) Time of the sigmoid fit
        # latency = x0 - ap_onset_time
        #
        # # Collect Results
        # # [Rec date, Rec name, AP number, min. Force, max. Force, delta Force, Steepness, Latency]
        # results.append(
        #     [rec_date,
        #      rec_name,
        #      ap,
        #      force_min,
        #      force_max,
        #      force_max-force_min,
        #      ap_onset_time,
        #      latency
        #      ]
        # )
        # print(f'AP: {ap} at {ap_onset_time:.3f} secs:')
        # print(f'Force increases from: {force_min:.4f} ({force_min_err:.4f}) '
        #       f'to {force_max:.4f} ({force_max_err:.4f}), '
        #       f'delta = {force_max-force_min:.4f}, '
        #       f'steepness= {steepness:.4f} ({steepness_err:.4f})')
        # print('')

        axs[1].plot(xdata, y_fit, 'b', lw=2.5, alpha=0.5)
    fig.savefig(base_dir / f'{rec_date}_{rec_name}_sigmoid_fit.png', dpi=300)
    plt.show()
else:
    fig.savefig(base_dir / f'{rec_date}_{rec_name}_overview.png', dpi=300)
    plt.show()

# Collect Results
if aps_bool:
    results_df = pd.DataFrame(results, columns=[
        'Date', 'RecName', 'AP_Nr', 'minForce', 'maxForce', 'deltaForce', 'APtime', 'TimeConstant', 'Latency', 'ID'
    ])
    results_df.to_csv(base_dir / f'{rec_date}_{rec_name}_results.csv', index=False)
    print('Results stored to csv file!')

# embed()
# exit()
