import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sp

#%% Defines class for a pair of xarray lidar datasets
class Dual_Doppler_Processing:
    def __init__(self, lidar1, lidar2, range1, range2, config, time_step=1,ang_tol=0.25,max_time_diff=2):
        
        self.config = config
        
        #read an check angles
        if lidar1.elevation.std()<ang_tol:
            self.el1 = np.float64(lidar1.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 1 is not constant")
            
        if lidar2.elevation.std()<ang_tol:
            self.el2 = np.float64(lidar2.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 2 is not constant")
            
        if lidar1.azimuth.std()<ang_tol:
            self.az1 = np.float64(lidar1.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 1 is not constant")
            
        if lidar2.azimuth.std()<ang_tol:
            self.az2 = np.float64(lidar2.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 2 is not constant")
             
        self.rg1 = range1
        self.rg2 = range2
        
        # Reorganizes lidar1 dataset so that scanID is replaced with time
        self.l1 = lidar1
        self.l2 = lidar2
        
        # Defines uniform time distribution
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(time_step, "s"))
        tnum=(self.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        
        #calculate the difference between the interpolation time and the nearest available time
        tnum1=(self.l1.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        tnum1=tnum1.expand_dims({"range":self.l1.range})
        tnum1=tnum1.where(self.l1.qc_wind_speed==0)
        time_diff1=tnum1.interp(time=self.time,method="nearest")-tnum
        
        tnum2=(self.l2.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        tnum2=tnum2.expand_dims({"range":self.l2.range})
        tnum2=tnum2.where(self.l2.qc_wind_speed==0)
        time_diff2=tnum2.interp(time=self.time,method="nearest")-tnum
        
        #interpolate radial wind speed over uniform time distribution
        self.ws1_int = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff1)<max_time_diff)
        self.ws2_int = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff2)<max_time_diff)
        
        
        interp_limit = self.config["interp_limit"] # pixels
        
        # Interpolates missing data values for lidar1
        valid_mask1 = ~np.isnan(self.ws1_int)
        distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
        interp_mask1 = (np.isnan(self.ws1_int)) & (distance1 <= interp_limit)
        yy1, xx1 = np.indices(self.ws1_int.shape)
        points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
        values1 = self.ws1_int.values[valid_mask1]
        interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
        interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
        self.ws1_inpaint = self.ws1_int.values.copy()
        self.ws1_inpaint[interp_mask1] = interpolated_values1
        
        # Interpolates missing data values for lidar2
        valid_mask2 = ~np.isnan(self.ws2_int)
        distance2 = sp.ndimage.distance_transform_edt(~valid_mask2)
        interp_mask2 = (np.isnan(self.ws2_int)) & (distance2 <= interp_limit)
        yy2, xx2 = np.indices(self.ws2_int.shape)
        points2 = np.column_stack((yy2[valid_mask2], xx2[valid_mask2]))
        values2 = self.ws2_int.values[valid_mask2]
        interp_points2 = np.column_stack((yy2[interp_mask2], xx2[interp_mask2]))
        interpolated_values2 = sp.interpolate.griddata(points2, values2, interp_points2, method='linear')
        self.ws2_inpaint = self.ws2_int.values.copy()
        self.ws2_inpaint[interp_mask2] = interpolated_values2
        
    # Finds wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        # Defines transformation matrix
        row1 = [self.cosd(self.el1) * self.sind(self.az1), self.cosd(self.el1) * self.cosd(self.az1)]
        row2 = [self.cosd(self.el2) * self.sind(self.az2), self.cosd(self.el2) * self.cosd(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.interp(range=self.rg1)
        rv2 = self.ws2_int.interp(range=self.rg2)
        rv = np.array([rv1, rv2])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel[0,:],wind_vel[1,:]
    
    # Finds average wind velocities
    # Returns 2D vector
    def av_wind_velocity(self):      
        # Defines transformation matrix
        row1 = [self.cosd(self.el1) * self.sind(self.az1), self.cosd(self.el1) * self.cosd(self.az1)]
        row2 = [self.cosd(self.el2) * self.sind(self.az2), self.cosd(self.el2) * self.cosd(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.sel(range=self.rg1).mean()
        rv2 = self.ws2_int.sel(range=self.rg2).mean()
        rv = np.array([rv1, rv2])
        
        # Solves for average wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel_av = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel_av
    
    # Finds wind speeds
    # Returns 1D array
    def wind_speed_direction(self,u=None,v=None):
        if u is None and v is None:
            u,v = self.wind_velocity()
            
        ws = np.hypot(u,v)
        wd = (270-np.degrees(np.arctan2(v,u)))%360
        
        return ws, wd
    
    # Plots radial wind speeds across a series of ranges over time
    # def plot_radial_speed(self):      
        # # Creates heatmaps of time and range against radial speed
        # fig1 = plt.figure(figsize=(15, 8))
        # ax1 = fig1.add_subplot(2,1,1)
        # plt.pcolor(self.time, self.ws1_int.range,self.ws1_int,cmap="plasma")
        # ax1.axhline(self.rg1,"--k")
        # 
        # ax1.set_xlabel("Time (UTC)")
        # ax1.set_ylabel("Range [m]")
        
        # plt.colorbar(label=f"Radial Wind Speed [m s$^{-1}$]")
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        
        # # Creates heatmap of time and range against radial speed for lidar2
        # fig2 = plt.figure(figsize=(15, 5))
        # ax2 = fig2.subplots()
        # #image2 = ax2.pcolormesh(self.time, range_limit2, radial_speeds2)
        # range_T2 = np.array(list(zip(range_limit2.values)))
        # image2 = ax2.pcolormesh(self.time, range_limit2, int_f2((range_T2, self.time)))
        # ax2.set_title("Lidar 2 Radial Wind Speed")
        # ax2.set_xlabel("Time (UTC)")
        # ax2.set_ylabel("Range (m)")
        # fig2.colorbar(image2, label="Radial Wind Speed (m/s)")
        # plt.show()
    
    # # Plots radial wind speeds at intersection point over time
    # def plot_radial_intersect(self):
    #     # Interpolates data over time distribution
    #     rv1 = self.ws1_int.sel(range=self.rg1)
    #     rv2 = self.ws2_int.sel(range=self.rg2)
        
    #     # Creates plot of time against lidar1 and lidar2 radial speeds
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, rv1, ".", label="Lidar 1")
    #     ax.plot(self.time, rv2, ".", label="Lidar 2")
    #     ax.grid(visible=True)
    #     ax.legend()
    #     ax.set_title("Radial Wind Speed at Intersection")
    #     ax.set_xlabel("Time (UTC)")
    #     ax.set_ylabel("Radial Wind Speed (m/s)")
    #     plt.show()
    
    # Plots wind velocity components at intersection point over time
    def plot_velocities(self,u=None,v=None):  
        
        date=str(self.time[0])[:10]
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 15))
        
        # Creates radial speed heatmaps
        rad1 = ax1.pcolormesh(self.time, self.ws1_int.range, self.ws1_int, cmap="plasma")
        fig.colorbar(rad1, label=f"Radial Wind Speed [m s$^{-1}$]")
        ax1.set_ylim([0, self.rg1+300])
        ax1.set_title(f"Lidar 1 Radial Wind Speed on {date}")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range (m)")
        rad2 = ax2.pcolormesh(self.time, self.ws2_int.range, self.ws2_int, cmap="plasma")
        fig.colorbar(rad2, label=f"Radial Wind Speed [m s$^{-1}$]")
        ax2.set_ylim([0, self.rg2+300])
        ax2.set_title(f"Lidar 2 Radial Wind Speed on {date}")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range (m)")
        
        # Creates inpainted radial speed heatmaps
        inpaint1 = ax3.pcolormesh(self.time, self.ws1_int.range, self.ws1_inpaint, cmap="plasma")
        fig.colorbar(inpaint1, label=f"Radial Wind Speed [m s$^{-1}$]")
        ax3.set_ylim([0, self.rg1+300])
        ax3.set_title(f"Inpainted Lidar 1 Radial Wind Speed on {date}")
        ax3.set_xlabel("Time (UTC)")
        ax3.set_ylabel("Range (m)")
        inpaint2 = ax4.pcolormesh(self.time, self.ws2_int.range, self.ws2_inpaint, cmap="plasma")
        fig.colorbar(inpaint2, label=f"Radial Wind Speed [m s$^{-1}$]")
        ax4.set_ylim([0, self.rg2+300])
        ax4.set_title(f"Inpainted Lidar 2 Radial Wind Speed on {date}")
        ax4.set_xlabel("Time (UTC)")
        ax4.set_ylabel("Range (m)")
        
        if u is None or v is None:
            u,v = self.wind_velocity()
        
        ws,wd=self.wind_speed_direction(u,v)
        
        # Creates plot of time against u and v component velocities
        #fig = plt.figure(figsize=(15, 5))
        #ax = fig.add_subplot(2,1,1)
        ax5.plot(self.time, u, ".b", label="U")
        ax5.plot(self.time, v, ".r", label="V")
        ax5.plot(self.time, ws, ".k", label="Wind speed")
        ax5.grid(visible=True)
        ax5.legend()
        ax5.set_title(f"Wind velocity on {date}")
        ax5.set_xlabel("Time (UTC)")
        ax5.set_ylabel(r"Wind velocity [m s$^{-1}$]")
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        
        #ax = fig.add_subplot(2,1,2)
        ax6.plot(self.time, wd, ".k")
        ax6.grid(visible=True)
        ax6.set_xlabel("Time (UTC)")
        ax6.set_ylabel(r"Wind direction [$^\circ$C]")
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.tight_layout()
        
        return fig
    
    # Plots wind speed at intersection point over time
    # def plot_speed(self):
    #     speeds = self.wind_speed().transpose()
        
    #     # Creates plot of time against wind speed
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, speeds, ".")
    #     ax.grid(visible=True)
    #     ax.set_title("Wind Speed")
    #     ax.set_xlabel("Time (UTC)")
    #     ax.set_ylabel("Wind Speed (m/s)")
    #     plt.show()
    
    # # Plots wind direction at intersection point over time
    # # Directions measured in degrees
    # def plot_direction(self):        
    #     wind_comp = self.wind_velocity()
        
    #     # Calculates wind direction angles from component data
    #     angles = np.arctan2(wind_comp[[1],:], wind_comp[[0],:])
    #     angles = (270 - angles) % 360
        
    #     # Plots time against wind direction
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, angles.transpose(), ".")
    #     ax.set_xlabel("Time(UTC)")
    #     ax.set_ylabel("Wind Direction (degrees)")
    #     ax.set_title("Wind Direction")
    #     ax.grid(visible=True)
    #     plt.show()
    
    # Plots correlation coefficient over a series of ranges
    def plot_correlation(self):
        #detrended RWS
        ws1_avg=self.ws1_int.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws2_avg=self.ws2_int.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws1_det=self.ws1_int-ws1_avg
        ws2_det=self.ws2_int-ws2_avg
        
        # Convert radial speed arrays into 3D
        rs1 = np.tile(ws1_det.values[:, np.newaxis, :], (1, len(ws2_det.range), 1))
        rs2 = np.tile(ws2_det.values[np.newaxis, :, :], (len(ws1_det.range), 1, 1))
        
        # Select real values
        real_mask = ~np.isnan(rs1) & ~np.isnan(rs2)
        
        # Calculate mean of time series and replace nan
        rs1[~real_mask] = np.nan
        rs2[~real_mask] = np.nan
        rs1_demean = rs1 - np.nanmean(rs1, axis=2, keepdims=True)
        rs2_demean = rs2 - np.nanmean(rs2, axis=2, keepdims=True)
        
        # Calculate covariance
        cov = np.nansum(rs1_demean * rs2_demean, axis=2)
        
        # Calculate product of standard deviations
        rs1_sqsum = np.nansum(rs1_demean**2, axis=2)
        rs2_sqsum = np.nansum(rs2_demean**2, axis=2)
        stand_devs = np.sqrt(rs1_sqsum * rs2_sqsum)
        
        # Calculates pearson correlation coefficient. Ignores undefined results
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = cov / stand_devs
        
        # Filters out ranges with less than two real data points
        valid_counts = np.sum(real_mask, axis=2)
        corr[valid_counts < self.config["correlation_valid"]] = np.nan
        
        # Creates heatmap of ranges against correlation coefficient
        date=str(self.time[0])[:10]
        fig = plt.figure()
        ax = fig.subplots()
        plt.pcolormesh(self.ws1_int.range,self.ws2_int.range, corr.T,cmap='seismic',vmin=-1,vmax=1)
        ax.set_title(f"Correlation Coefficient on {date}")
        ax.set_xlabel("Lidar 1 Range (m)")
        ax.set_ylabel("Lidar 2 Range (m)")

        ax.axvline(self.rg1, color="k")
        ax.axhline(self.rg2, color="k")
        plt.colorbar(label='Correlation coefficient')
        plt.xlim([0,self.rg1+300])
        plt.ylim([0,self.rg2+300])
        plt.grid()
        
        return fig
    
    def cosd(self,angle):
        return math.cos(math.radians(angle))
     
    def sind(self,angle):
        return math.sin(math.radians(angle))
    
# #%%
# lidar1 = "C:/Users/zuribe/Documents/GitHub/awaken_dual_doppler/data/awaken/sa5.lidar.z03.b0/sa5.lidar.z03.b0.20230727.002005.user5.vt.nc"
# lidar2 = "C:/Users/zuribe/Documents/GitHub/awaken_dual_doppler/data/awaken/arm.lidar.sgp_s4.rhi2.b2/sgpdlrhi2S4.b2.20230727.002007.vt.nc"
# elevation1 = 1.55 * (math.pi/180)
# elevation2 = 2.33 * (math.pi/180)
# azimuth1 = 309.98 * (math.pi/180)
# azimuth2 = 356.19 * (math.pi/180)
# range1 = 2925 #actual range from A5: 2929
# range2 = 1875 #actual range from A1: 1861

# Wind_Data = Dual_Doppler_Processing(xr.open_dataset(lidar1), xr.open_dataset(lidar2), range1, range2)

# #wind_vel=Wind_Data.wind_velocity()
# #fig_corr=Wind_Data.plot_correlation()
# Wind_Data.plot_radial_speed()