import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sp
#%% Defines class for a pair of xarray lidar datasets
# Assumes datasets have equal number of ranges
class Lidar_Dataset:
    def __init__(self, lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2):
        self.el1 = elevation1
        self.el2 = elevation2
        self.az1 = azimuth1
        self.az2 = azimuth2
        self.rg1 = range1
        self.rg2 = range2
        
        # Reorganizes lidar1 dataset so that scanID is replaced with time
        self.l1 = xr.open_dataset(lidar1)
        self.l1 = self.l1.assign_coords(scanID=self.l1.time)
        self.l1 = self.l1.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        # Reorganizes lidar2 dataset so that scanID is replaced with time
        self.l2 = xr.open_dataset(lidar2)
        self.l2 = self.l2.assign_coords(scanID=self.l2.time)
        self.l2 = self.l2.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        # Defines uniform time distribution
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(1, "s"))
        # Interpolates wind speed over uniform time distribution
        self.ws1_int = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).interp(time=self.time)
        self.ws2_int = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).interp(time=self.time)
        
    # Finds wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.sel(range=self.rg1)
        rv2 = self.ws2_int.sel(range=self.rg2)
        rv = np.array([rv1, rv2])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel
    
    # Finds average wind velocities
    # Returns 2D vector
    def av_wind_velocity(self):      
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
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
    def wind_speed(self):
        wind_comp = self.wind_velocity()
        ws = np.hypot(wind_comp[[0], :], wind_comp[[1], :])
        return ws
    
    # Plots radial wind speeds across a series of ranges over time
    # Ranges limited to specified ranges
    def plot_radial_speed(self):
        # Defines range and interpolation limit
        lim = (0, 100)
        range_limit1 = self.ws1_int.range[lim[0]:lim[1]]
        radial_speeds1 = self.ws1_int.values[lim[0]:lim[1]]
        range_limit2 = self.ws2_int.range[lim[0]:lim[1]]
        radial_speeds2 = self.ws2_int.values[lim[0]:lim[1]]
        interp_limit = 1 # pixels
        
        # Interpolates missing data values for lidar1
        # Remove these lines to skip interpolation
        valid_mask1 = ~np.isnan(radial_speeds1)
        distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
        interp_mask1 = (np.isnan(radial_speeds1)) & (distance1 <= interp_limit)
        yy1, xx1 = np.indices(radial_speeds1.shape)
        points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
        values1 = radial_speeds1[valid_mask1]
        interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
        interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
        result1 = radial_speeds1.copy()
        result1[interp_mask1] = interpolated_values1
        
        # Interpolates missing data values for lidar2
        # Remove these lines to skip interpolation
        valid_mask2 = ~np.isnan(radial_speeds2)
        distance2 = sp.ndimage.distance_transform_edt(~valid_mask2)
        interp_mask2 = (np.isnan(radial_speeds2)) & (distance2 <= interp_limit)
        yy2, xx2 = np.indices(radial_speeds2.shape)
        points2 = np.column_stack((yy2[valid_mask2], xx2[valid_mask2]))
        values2 = radial_speeds2[valid_mask2]
        interp_points2 = np.column_stack((yy2[interp_mask2], xx2[interp_mask2]))
        interpolated_values2 = sp.interpolate.griddata(points2, values2, interp_points2, method='linear')
        result2 = radial_speeds2.copy()
        result2[interp_mask2] = interpolated_values2
        
        # Creates heatmap of time and range against radial speed for lidar1
        fig1 = plt.figure(figsize=(15, 5))
        ax1 = fig1.subplots()
        image1 = ax1.pcolormesh(self.time, range_limit1, result1)
        ax1.set_title("Lidar 1 Radial Wind Speed")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range (m)")
        fig1.colorbar(image1, label="Radial Wind Speed (m/s)")
        plt.show()
        
        # Creates heatmap of time and range against radial speed for lidar2
        fig2 = plt.figure(figsize=(15, 5))
        ax2 = fig2.subplots()
        image2 = ax2.pcolormesh(self.time, range_limit2, result2)
        ax2.set_title("Lidar 2 Radial Wind Speed")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range (m)")
        fig2.colorbar(image2, label="Radial Wind Speed (m/s)")
        plt.show()
    
    # Plots radial wind speeds at intersection point over time
    def plot_radial_intersect(self):
        # Selects intersetction range
        rv1 = self.ws1_int.sel(range=self.rg1)
        rv2 = self.ws2_int.sel(range=self.rg2)
        
        # Creates plot of time against lidar1 and lidar2 radial speeds
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, rv1, ".", label="Lidar 1")
        ax.plot(self.time, rv2, ".", label="Lidar 2")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Radial Wind Speed at Intersection")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Radial Wind Speed (m/s)")
        plt.show()
    
    # Plots wind velocity components at intersection point over time
    def plot_components(self):  
        # Generates wind component data
        wind_comp = self.wind_velocity()
        
        # Creates plot of time against u and v component velocities
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, wind_comp[[0],:].transpose(), ".", label="U")
        ax.plot(self.time, wind_comp[[1],:].transpose(), ".", label="V")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Wind Components")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plots wind speed at intersection point over time
    def plot_speed(self):
        # Generates wind speed data
        speeds = self.wind_speed().transpose()
        
        # Creates plot of time against wind speed
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, speeds, ".")
        ax.grid(visible=True)
        ax.set_title("Wind Speed")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plots wind direction at intersection point over time
    # Directions measured in degrees
    def plot_direction(self):        
        # Generates wind component data
        wind_comp = self.wind_velocity()
        
        # Calculates wind direction angles from component data
        angles = np.arctan2(wind_comp[[1],:], wind_comp[[0],:])
        angles = (270 - angles) % 360
        
        # Plots time against wind direction
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, angles.transpose(), ".")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Direction (degrees)")
        ax.set_title("Wind Direction")
        ax.grid(visible=True)
        plt.show()
    
    # Plots correlation coefficient over a series of ranges
    # Ranges limited to specified ranges
    def plot_correlation(self):
        # Defines range limit
        lim1 = (60, 110)
        lim2 = (40, 70)
        range_limit1 = self.ws1_int.range[lim1[0]:lim1[1]]
        radial_speeds1 = self.ws1_int.values[lim1[0]:lim1[1]]
        range_limit2 = self.ws2_int.range[lim2[0]:lim2[1]]
        radial_speeds2 = self.ws2_int.values[lim2[0]:lim2[1]]
        
        # Convert radial speed arrays into 3D
        rs1 = np.tile(radial_speeds1[:, np.newaxis, :], (1, len(range_limit2), 1))
        rs2 = np.tile(radial_speeds2[np.newaxis, :, :], (len(range_limit1), 1, 1))
        
        # Select real values
        real_mask = ~np.isnan(rs1) & ~np.isnan(rs2)
        
        # Calculate mean of time series and replace nan
        rs1_demean = rs1 - np.nanmean(rs1, axis=2, keepdims=True)
        rs2_demean = rs2 - np.nanmean(rs2, axis=2, keepdims=True)
        rs1_demean[~real_mask] = 0
        rs2_demean[~real_mask] = 0
        
        # Calculate covariance
        cov = np.sum(rs1_demean * rs2_demean, axis=2)
        
        # Calculate product of standard deviations
        rs1_sqsum = np.sum(rs1_demean**2, axis=2)
        rs2_sqsum = np.sum(rs2_demean**2, axis=2)
        stand_devs = np.sqrt(rs1_sqsum * rs2_sqsum)
        
        # Calculates pearson correlation coefficient. Ignores undefined results
        with np.errstate(invalid='ignore', divide='ignore'):
            coef = cov / stand_devs
        
        # Filters out ranges with less than two real data points
        valid_counts = np.sum(real_mask, axis=2)
        coef[valid_counts < 2] = np.nan
     
        # This version of the function produced a nearly identical plot
        """
        # Builds empty correlation coefficient matrix
        coef = np.zeros((len(range_limit1), len(range_limit2)))
        # Generate correlation coefficient data
        for r1 in range(len(range_limit1)):
            for r2 in range(len(range_limit2)):
                rs1 = radial_speeds1[r1]
                rs2 = radial_speeds2[r2]
                real_rs1 = rs1[~np.isnan(rs1 + rs2)]
                real_rs2 = rs2[~np.isnan(rs1 + rs2)]
                corr = np.corrcoef(real_rs1, real_rs2)[0, 1]
                coef[r1, r2] = corr
        """
        
        # Creates heatmap of ranges against correlation coefficient
        fig = plt.figure()
        ax = fig.subplots()
        image = ax.pcolormesh(range_limit2, range_limit1, coef)
        ax.set_title("Correlation Coefficient")
        ax.set_xlabel("Lidar 2 Range (m)")
        ax.set_ylabel("Lidar 1 Range (m)")
        fig.colorbar(image, label="Correlation Coefficient")
        ax.axvline(self.rg2, color="r")
        ax.axhline(self.rg1, color="r")
        plt.show()
#%%
lidar1 = "sa5.lidar.z03.b0.20230726.002006.user5.vt.nc"
lidar2 = "sgpdlrhi2S4.b2.20230726.002007.vt.nc"
elevation1 = 1.55 * (math.pi/180)
elevation2 = 2.33 * (math.pi/180)
azimuth1 = 309.98 * (math.pi/180)
azimuth2 = 356.19 * (math.pi/180)
range1 = 2925
range2 = 1875

Wind_Data = Lidar_Dataset(lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2)