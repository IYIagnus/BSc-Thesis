import numpy as np
import matplotlib.pyplot as plt
import PlotIVS
import DataProcessing
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D

def plotByMoneyness(df, k, dim, subject_list, ylabel):
    t = dim[0]*dim[1]
    split_ttm = DataProcessing.weightedSplit(df, t, 'ttm')
    split_moneyness = []
    for ttm in split_ttm:
        split_moneyness.append(DataProcessing.weightedSplit(ttm, k, 'moneyness'))
      
    fig = plt.figure()
    subplots = []
    plot_list = []
    axins_list = []
    
    for subject in subject_list:
        for ttm in split_moneyness:
            temp = []
            for m in ttm:
                temp.append([m['moneyness'].median(), m[subject].mean()])
            plot_list.append(np.array(temp))
         
    height = dim[1]
    for i in range(t):
        subplots.append(fig.add_subplot(int(str(height)+str(dim[0])+str(i+1))))
#        axins_list.append(zoomed_inset_axes(subplots[i], 2, loc=1))
        subplots[i].set_title('Maturity: '+str(int(split_ttm[i]['ttm'].min()*365))+' - '+str(int(split_ttm[i]['ttm'].max()*365))+' days')
        subplots[i].set_xlabel('moneyness')
        subplots[i].set_ylabel(ylabel)
    
    
    count = 0
    for j in range(len(subject_list)):
#        xmin = 1.025
        for l in range(len(subplots)):
            subplots[l].plot(plot_list[count][:, 0], plot_list[count][:, 1], label=str(subject_list[j]))
#            axins_list[l].plot(plot_list[count][:, 0], plot_list[count][:, 1])
#            axins_list[l].set_xlim(xmin, plot_list[count][:, 0].max()+0.01) # apply the x-limits
#            axins_list[l].set_ylim(plot_list[count][:, 1].min()-0.01, plot_list[count][:, 1][-1]+0.02) # apply the y-limits
#            axins_list[l].xaxis.set_visible(False)
#            axins_list[l].yaxis.set_visible(False)
            
            count += 1
        
#    for p in range(len(axins_list)):
#        mark_inset(subplots[p], axins_list[p], loc1=3, loc2=1, fc="none", ec="0.5")
#            
#    axins_list[2].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[3].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[4].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[5].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[6].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[7].set_xlim(1., 1.1) # apply the x-limits
#    axins_list[8].set_xlim(1., 1.2) # apply the x-limits


    plt.show()    
    
def plotByTTM(df, t, dim, subject_list, ylabel):
    k = dim[0]*dim[1]
    split_moneyness = DataProcessing.weightedSplit(df, k, 'moneyness')
    split_ttm = []
    for moneyness in split_moneyness:
        split_ttm.append(DataProcessing.weightedSplit(moneyness, t, 'ttm'))
      
    fig = plt.figure()
    for subject in subject_list:
        plot_list = []
        for moneyness in split_ttm:
            temp = []
            for m in moneyness:
                temp.append([m['ttm'].median(), m[subject].mean()])
            plot_list.append(np.array(temp))
         
        count = 1
        subplots = []
        height = dim[1]
        for i in range(len(plot_list)):
            subplots.append(fig.add_subplot(int(str(height)+str(dim[0])+str(count))))
            count += 1
            
        for j in range(len(subplots)):
            subplots[j].plot(plot_list[j][:, 0], plot_list[j][:, 1])
            subplots[j].set_title('Moneyness: '+str(round(split_moneyness[j]['moneyness'].min(), 3))+' - '+str(round(split_moneyness[j]['moneyness'].max(), 3)))
            subplots[j].set_xlabel('time-to-maturity')
            subplots[j].set_ylabel(ylabel)
                
    plt.show()


def getGridForIVS(df, k, t, subject):
    split_ttm = DataProcessing.unweightedSplit(df, t, 'ttm')
    split_moneyness = []
    
    for ttm in split_ttm:
        split_moneyness.append(DataProcessing.unweightedSplit(ttm, k, 'moneyness'))
        
    grid = []
    for ttm in split_moneyness:
        temp = []
        for m in ttm:
            temp.append([m['moneyness'].median(), m[subject].mean()])
        grid.append(temp)
    
    output = []
    
    for i in range(t):
        for j in range(k):
            output.append(grid[i][j] + [split_ttm[i]['ttm'].median()])
        
    return np.array(output)

def plotImpliedVolSurf(df, k, t, subjects):
    
    df = df[df.moneyness > 0.8]
    df = df[df.ttm > 0.2]
    
    market_grid = getGridForIVS(df, k, t, 'IV')
    grids = []
    
    for i in subjects:
        grids.append(getGridForIVS(df, k, t, i))
        
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    surf1 = ax1.plot_trisurf(grids[0][:, 0], grids[0][:, 2], grids[0][:, 1], 
                             cmap=cm.coolwarm, linewidth=0.1, 
                             vmin=grids[0][:, 1].min(), vmax=grids[0][:, 1].max())
    
    surf2 = ax2.plot_trisurf(grids[1][:, 0], grids[1][:, 2], grids[1][:, 1], 
                             cmap=cm.coolwarm, linewidth=0.1, 
                             vmin=grids[1][:, 1].min(), vmax=grids[1][:, 1].max())
    
    mark1 = ax1.plot_trisurf(market_grid[:, 0], market_grid[:, 2], 
                             market_grid[:, 1]+0.05, 
                             cmap=cm.Oranges, linewidth=0.1, 
                             vmin=market_grid[:, 1].min(), 
                             vmax=market_grid[:, 1].max())
    
    mark2 = ax2.plot_trisurf(market_grid[:, 0], market_grid[:, 2], 
                             market_grid[:, 1]+0.05, 
                             cmap=cm.Oranges, linewidth=0.1, 
                             vmin=market_grid[:, 1].min(), 
                             vmax=market_grid[:, 1].max())
    
    plt.show()