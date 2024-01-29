import pyabf
import pandas as pd
from pathlib import Path
from IPython import embed

# SETTINGS
abf_file_dir = Path('C:/UniFreiburg/VenusFlyTrap/recordings/20231030/20231030_ephys_data_12.abf')
save_dir = Path(f'{abf_file_dir.as_posix()[:-4]}.csv')

# Open abf file
abf = pyabf.ABF(abf_file_dir.as_posix())

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GET CHANNEL 1
channel_01_x = abf.sweepX
channel_01_y = abf.sweepY

fr = 1000
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GET CHANNEL 2
abf.setSweep(sweepNumber=0, channel=1)
channel_02_x = abf.sweepX
channel_02_y = abf.sweepY

df = pd.DataFrame()
df['Time'] = channel_01_x
df['Voltage'] = channel_01_y
df['Stimulus'] = channel_02_y

df.to_csv(save_dir.as_posix())
print('Data Converted and Stored to HDD')
