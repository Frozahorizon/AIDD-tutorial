import os
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

################################################
# Create a new folder to save csv file and figures
warnings.filterwarnings('ignore')

try:
    os.makedirs('output_file')
    print('Creating the output file folder successfully')
except FileExistsError:
    pass

try:
    os.makedirs('output_figure')
    print('Creating the output figure folder successfully')
except FileExistsError:
    pass

################################################
# global variable
file_path = './dataset/DOWNLOAD-xo9eE_YcwPUPS1uBnt8j5kfkIDuQ-KTjiRRBeJ-ps4M=.csv'
normal_method = 'Standard'
# normal_method = 'MinMax'

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

################################################
# input dataset
data = pd.read_csv(file_path)
# data = pd.read_excel(file_path)  # if the data type is xlsx

print(data.columns)  # print the base information of dataset

data = data.loc[:, ['Smiles', 'pChEMBL Value']]  # save the structure and activity values
data = data.dropna()  # remove row with missing value
data.index = range(len(data))
names = data.columns
print(data.shape)

smiles = data.iloc[:, 0]  # fetch the molecular structure with smiles format
df_smiles = pd.DataFrame(smiles)

print(smiles.head())

activity = data.iloc[:, 1]  # fetch the molecular activity values with pIC50
activity = activity.astype(np.float32)
df_activity = pd.DataFrame(activity)
print(df_activity.head())

binary_activity = []
for a in activity:
    if a >= 6.0:
        binary_activity.append(1.0)
    else:
        binary_activity.append(0.0)

df_binary = pd.DataFrame(binary_activity, columns=['activity label'])
df_binary.to_csv('./output_file/binary_activity.csv', index=False)

if normal_method == 'Standard':
    scaler = StandardScaler()
    activity = scaler.fit_transform(df_activity)
elif normal_method == 'MinMax':
    scaler = MinMaxScaler(feature_range=(0, 1))
    activity = scaler.fit_transform(activity)
else:
    print('Using wrong method of normalization')
    print('This script supports the StandardScaler and MinMaxScaler')

df1 = pd.DataFrame(smiles)
df2 = pd.DataFrame(activity)

df = pd.concat([df1, df2], axis=1)
df.columns = names
df.to_csv('./output_file/clean.csv', index=False)
print('The data preprocessing has done')

end_time = datetime.datetime.now()
print('End running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))