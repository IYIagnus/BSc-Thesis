import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

from scipy.interpolate import interp2d, griddata
from matplotlib import cm

def plotIVS(moneyness, ttm, IV):
    
    grid_x, grid_y = np.mgrid[moneyness.min():moneyness.max():100j, ttm.min():ttm.max():100j]
    
    points = (moneyness, ttm)
    values = IV
    
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm, 
                    vmin=np.nanmin(grid_z), vmax=np.nanmax(grid_z), 
                    antialiased=True)
    
def compareIVS(cord1, cord2):
    
    grid_x1, grid_y1 = np.mgrid[cord1[0].min():cord1[0].max():200j, cord1[1].min():cord1[1].max():200j]
    grid_x2, grid_y2 = np.mgrid[cord2[0].min():cord2[0].max():200j, cord2[1].min():cord2[1].max():200j]
    
    points1 = (cord1[0], cord1[1])
    points2 = (cord2[0], cord2[1])
    values1 = cord1[2]
    values2 = cord2[2]
    
    grid_z1 = griddata(points1, values1, (grid_x1, grid_y1), method='linear')
    grid_z2 = griddata(points2, values2, (grid_x2, grid_y2), method='linear')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(grid_x1, grid_y1, grid_z1, cmap=cm.coolwarm, 
                    vmin=np.nanmin(grid_z1), vmax=np.nanmax(grid_z1), 
                    antialiased=True)

    ax.plot_surface(grid_x2, grid_y2, grid_z2, cmap=cm.Spectral, 
                    vmin=np.nanmin(grid_z2), vmax=np.nanmax(grid_z2), 
                    antialiased=True)
    
def plotIVSFromDF(df):
        
    x, y, z = np.array(df["moneyness"]), np.array(df["ttm"]), np.array(df["IV"])
    
    plotIVS(x, y, z)

def compareIVSFromDF(df, subjects):    
    
    x1, y1, z1 = np.array(df["moneyness"]), np.array(df["ttm"]), np.array(df[subjects[0]])
    
    x2, y2, z2 = np.array(df["moneyness"]), np.array(df["ttm"]), np.array(df[subjects[1]]+0.2)
    
    compareIVS((x1, y1, z1), (x2, y2, z2))
    

    
