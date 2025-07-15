import matplotlib.pyplot as plt
import synthetic_processing as lxr
import numpy as np
#%% Set scan and range limits
scans = lxr.lidar1.scanID[0:]
ranges = lxr.lidar1.range[0:100]
#%% Generate lidar1 data
wind_data1 = np.zeros((len(ranges), len(scans)))

for i in range(0, len(ranges)):
    for j in range(0, len(scans)):
        wind_data1[i, j] = lxr.lidar1.wind_speed.where(lxr.lidar1.qc_wind_speed==0).sel(range=ranges[i], scanID=scans[j])
#%% Plot lidar1
fig1 = plt.figure()
ax1 = fig1.subplots()

image1 = ax1.pcolormesh(scans, ranges, wind_data1)
ax1.set_title("A5 Radial Wind Speed")
ax1.set_xlabel("scanID")
ax1.set_ylabel("Range")
fig1.colorbar(image1, label="Radial Wind Speed")
plt.show()
#%% Generate lidar2 data
wind_data2 = np.zeros((len(ranges), len(scans)))

for i in range(0, len(ranges)):
    for j in range(0, len(scans)):
        wind_data2[i, j] = lxr.lidar2.wind_speed.where(lxr.lidar2.qc_wind_speed==0).sel(range=ranges[i], scanID=scans[j])
#%% Plot lidar2
fig2 = plt.figure()
ax2 = fig2.subplots()

image2 = ax2.pcolormesh(scans, ranges, wind_data2)
ax2.set_title("A1 Radial Wind Speed")
ax2.set_xlabel("scanID")
ax2.set_ylabel("Range")
fig2.colorbar(image2, label="Radial Wind Speed")
plt.show()
#%% Plot radial wind speeds at intersection point
fig3 = plt.figure()
ax3 = fig3.subplots()

A5_radial_speed = []
for i in range(0, len(scans)):
    rad_speed = lxr.lidar1.wind_speed.where(lxr.lidar1.qc_wind_speed==0).sel(range=lxr.range1, scanID=scans[i])
    A5_radial_speed.append(rad_speed)
    
A1_radial_speed = []
for i in range(0, len(scans)):
    rad_speed = lxr.lidar2.wind_speed.where(lxr.lidar2.qc_wind_speed==0).sel(range=lxr.range2, scanID=scans[i])
    A1_radial_speed.append(rad_speed)

ax3.plot(scans, A5_radial_speed, label="A5")
ax3.plot(scans, A1_radial_speed, label="A1")
ax3.grid(visible=True)
ax3.legend()
ax3.set_title("Radial Wind Speed at Intersection")
ax3.set_xlabel("scanID")
ax3.set_ylabel("Radial Wind Speed")
plt.show()