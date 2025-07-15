import matplotlib.pyplot as plt
import synthetic_processing as lxr
#%% Set scan limit and generate component data
scans = lxr.lidar1.scanID[0:]

u_wind = []
for i in scans:
    wv = lxr.wind_velocity(sc=i)
    u_wind.append(wv[0])

v_wind = []
for i in scans:
    wv = lxr.wind_velocity(sc=i)
    v_wind.append(wv[1])
#%% Generate wind speed data
wind = []
for i in scans:
    wv = lxr.wind_velocity(sc=i)
    ws = lxr.wind_speed(wv)
    wind.append(ws)
#%% Plot components and wind speed
fig1 = plt.figure()
ax1 = fig1.subplots()
fig2 = plt.figure()
ax2 = fig2.subplots()

ax1.plot(scans, u_wind, label="u")
ax1.plot(scans, v_wind, label="v")
ax1.grid(visible=True)
ax1.legend()
ax1.set_title("Wind Components")
ax1.set_xlabel("scanID")
ax1.set_ylabel("Wind Speed")

ax2.plot(scans, wind)
ax2.grid(visible=True)
ax2.set_title("Wind Speed")
ax2.set_xlabel("scanID")
ax2.set_ylabel("Wind Speed")

plt.show()