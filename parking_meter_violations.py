# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:35:29 2018

@author: David
"""
import os
cwd = os.getcwd()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


mysql_dir = "C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\"


def timeConversion(series):
    """Convert 0403P to 04:03 PM and so on."""
    series = series.astype("str")
    times = {t:":".join((t[0:2],t[2:4]))+str(" "+t[-1]+"M") for t in series.unique()}
    return series.map(times)


def datetimeConversion(series):
    series = series.astype("str")
    dates = {date:pd.to_datetime(date, errors="coerce") for date in series.unique()}
    return series.map(dates)


def clean_meter_data():
    """Process data to format times etc."""
    path = mysql_dir + "meter.csv" #CSV file containing violation code counts
    names = ['Issue_Date', 'Violation_Code', 'Violation_Time', 'Violation_County', 
             'Violation_In_Front_Of_Or_Opposite', 'House_Number', 'Street_Name', 
             'Days_Parking_In_Effect', 'From_Hours_In_Effect', 'To_Hours_In_Effect']
    data = pd.read_csv(path, header=None, names=names) # import 
    data = data.dropna().reset_index(drop=True)
    data["Violation_Time"] = timeConversion(data["Violation_Time"])
    data["From_Hours_In_Effect"] = timeConversion(data["From_Hours_In_Effect"])
    data["To_Hours_In_Effect"] = timeConversion(data["To_Hours_In_Effect"])
    data["Datetime"] = data["Issue_Date"]+str(" ")+data["Violation_Time"]
    data["Datetime"] = datetimeConversion(data["Datetime"])
    data = data.dropna().reset_index(drop=True)
    data["Weekday"] = data["Datetime"].apply(lambda x: x.weekday())
 
    
    def exportData(data):
        """Quick export to save processed CSV"""
        header = data.columns.tolist()
        data.to_csv(cwd + "/meter_cleaned.csv", sep=',', columns=header)
        return
    
    exportData(data)
    return


def importData():
    data = pd.read_csv(cwd + "/meter_cleaned.csv", header=0, index_col=0)
    return data


def ZCA_normalize(array):
    stddev = np.std(array)
    avg = np.mean(array)
    array = (array-avg)/stddev
    return array


def meter_weekday_plot(data,save=False):
    """Bar chart to show meter violations per weekday."""
    plt.figure(figsize=(10,7.5))
    plt.title("Meter Violations By Weekday", fontsize=20)
    plt.xlabel("Weekday", fontsize=16)
    plt.ylabel("ZCA Count", fontsize=16)
    
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
      
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    
    counts = data["Weekday"].value_counts().sort_index()
    x = counts.index
    idx = np.arange(len(x))
    y = counts.values
    y = ZCA_normalize(y)
    x_labels = ["Mon","Tues","Wed","Thurs","Fri","Sat","Sun"]
    plt.xticks(idx, x_labels, fontsize=14)  
    plt.bar(idx, y, width=0.55, color="navy", alpha=0.80)
    plt.xlim([min(idx)-0.5,max(idx)+0.5])
    plt.ylim([-2.5,2.5])
    plt.hlines(y=0,
               xmin=min(x)-0.5,
               xmax=max(x)+0.5,
               linestyle='dashed')
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("meter_weekday_plot.png",format="png",dpi=300)
    plt.show()
    return


def meter_yearday_plot(data,save=False):
    """Meter violations by day of year."""
    plt.figure(figsize=(10,7.5))
    plt.title("Meter Violations By Day of Year", fontsize=20)
    plt.xlabel("Day of Year", fontsize=16)
    plt.ylabel("ZCA Count", fontsize=16)
    
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
      
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    
    try:
        data["Datetime"] = data["Datetime"].astype("datetime64")
    except:
        pass
    data["doy"] = data["Datetime"].dt.dayofyear
    counts = data["doy"].value_counts().sort_index()
    x = counts.index
    idx = np.arange(len(x))
    y = counts.values
    y = ZCA_normalize(y)
    y_rm = counts.rolling(7).mean() #rolling mean
    plt.plot(idx, ZCA_normalize(y_rm),'ro-') #ZCA rolling mean
    plt.bar(idx, y, width=1.0, color="red", alpha=0.30)
    plt.legend(("Rolling Mean","ZCA Count"))
    plt.xlim([min(idx)-0.5,max(idx)+0.5])
    plt.tight_layout()
    if save:
        plt.savefig("meter_doy_plot.png",format="png",dpi=300)
    plt.show()
    return


def meter_time_plot(data,save=False):
    """Plot overlapping distributions, violation_time vs. start_time"""
    plt.figure(figsize=(10,7.5))
    plt.title("Meter Violation Times", fontsize=20)
    plt.xlabel("24-Hour Time", fontsize=16)
    plt.ylabel("Normalized Intensity", fontsize=16)
    
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
      
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    
    y, x, _ = plt.hist(data["Time"].values, 
                       bins=100, alpha=0.35,
                       edgecolor='blue',
                       label='hist', normed=True)

    x=(x[1:]+x[:-1])/2

    def gauss(x,mu,sigma,A):
        return A*np.exp(-((x-mu)**2)/(1/(2*sigma**2)))
    
    def four_modes(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4):
        g1 = gauss(x,mu1,sigma1,A1)
        g2 = gauss(x,mu2,sigma2,A2)
        g3 = gauss(x,mu3,sigma3,A3)
        g4 = gauss(x,mu4,sigma4,A4)
        return g1+g2+g3+g4
    
    expected=[0.01]*12
    params,cov = curve_fit(four_modes,x,y,expected)
    sigma=np.sqrt(np.diag(cov))
    plt.plot(x,four_modes(x,*params),color="red",linewidth=3.0,label='model')
    plt.legend(("Mixed Gaussians","Histogram"), fontsize=12)
    plt.xlim([5.0,24.0])
    if save:
        plt.savefig("meter_violation_times.png", format="png",dpi=300)
    gmm = pd.DataFrame(data={'params':params,'sigma':sigma},
                       index=['mu1','sigma1','A1','mu2','sigma2','A2','mu3','sigma3','A3','mu4','sigma4','A4'])
    return gmm


if __name__=="__main__":
    #checkpoint 1
    #clean_meter_data()
    data = importData()
    
    meter_weekday_plot(data,save=True)
    """As expected, very unlikely on Sundays."""
    
    meter_yearday_plot(data,save=True) 
    """Transitions correspond to ~Jan. 24th, Jun. 24th, and Nov. 24th."""

    #Make time conversions for convenience
    data["Start_Time"] = datetimeConversion(data["From_Hours_In_Effect"])
    data = data[~data["Start_Time"].isnull()]
    data = data.reset_index(drop=True)
    data["Start_Time"] = data["Start_Time"].apply(lambda x: x.time().hour+x.time().minute/60.0)
    data["Time"] = data["Datetime"].apply(lambda x: x.time().hour+x.time().minute/60.0)
    data["delay"] = data["Time"]-data["Start_Time"]

    gmm = meter_time_plot(data,save=True)
    """4 Peaks: 10:00 AM, 1:44 PM, 5:00 PM, 9:00 PM"""