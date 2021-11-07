import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.signal as sg
from scipy.signal import butter,filtfilt

#print to 3 decimal points
np.set_printoptions(precision=3)

#create empty array to store +/- slope change points
df_slopeChange = []

#open and read file, set columns to x and y
data = pd.read_csv("A39_2_MD5MEM_4MPR_7680-9647-page-012.csv", sep=',', header=None, skiprows=1, names=['x','y'])
data_y = data[['y']]
frequency = np.array(data_y)
print(frequency)
data_x = data[['x']]
time = np.array(data_x)
#plt.plot(time,frequency)

#filter parameters
order = 2 #filter order
cutoff = 0.01 #cutoff

#create filter
b, a = sg.butter(order, cutoff, output='ba')

#apply filter
freqfilt = sg.filtfilt(b, a, frequency, axis=0)
print(freqfilt)

np.savetxt('filtered.csv', freqfilt, delimiter=',', fmt='%1.3f')

#set x and y
x = time
y = freqfilt

#collect total number of samples
lines = len(data)
print(lines)

#detect slope change
prior_slope = float(y[1] - y[0]) / (x[1] - x[0])

#if next slope is same +/- as prior slope continue, if change, append
for n in range(0, len(x)): 
    slope = float(y[n] - y[n - 1]) / (x[n] - x[n - 1])
    if slope > 0 and prior_slope > 0:
        continue
    if slope < 0 and prior_slope < 0:
        continue
    if slope > 0 and prior_slope < 0:
        df_slopeChange.append(n)
    if slope < 0 and prior_slope > 0:
        df_slopeChange.append(n)
    prior_slope = slope

#index of slope change points
df_slopeChange=np.array(df_slopeChange)

#collect actual x and y values of slope change points in filtered signal
y_filt = np.array(freqfilt[df_slopeChange])
x_filt = np.array(x[df_slopeChange])

#new array of xy for slopechange 
filtered_xy = np.column_stack((x_filt, y_filt))
filtered_xy = np.array(filtered_xy)
#print(filtered_xy)
#print(filtered_xy.ndim)
#print(filtered_xy.shape)

midpoints_x = []
midpoints_y =[]

for i in range(0, len(x_filt)-1):
    mid_x = ((x_filt[i] + x_filt[i+1])/2)
    mid_y = ((y_filt[i] + y_filt[i+1])/2)
    midpoints_x.append(mid_x)
    midpoints_y.append(mid_y)
    
midpoints_x = np.array(midpoints_x)
midpoints_y = np.array(midpoints_y)
midpoints = np.column_stack((midpoints_x,midpoints_y))

plt.plot(midpoints_x, midpoints_y, drawstyle='steps')
plt.plot(time, freqfilt, 'r-')
#plt.plot(x_filt, y_filt, 'o')
plt.show()

np.savetxt('slopechangepoints.csv', filtered_xy, fmt='%1.3f')


