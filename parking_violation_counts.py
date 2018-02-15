# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:35:29 2018

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files_dir = "C:\\Users\\David\\Desktop\\PythonCode\\Projects\\ParkingViolations\\files\\"
mysql_dir = "C:\\ProgramData\\MySQL\\MySQL Server 5.7\\Uploads\\"


"""First let's look at the most common parking violations!"""

def violation_code_descriptions():
    """Import violation codes and their descriptions."""
    path = files_dir + "STARS_VIOLATION_CODES.txt" #Text file containing code definitions
    data = pd.read_csv(path, header = None, names = ["Violation_Code", "Description"]) # import
    data["Description"] = data["Violation_Code"] #Copy column
    data["Violation_Code"] = data["Violation_Code"].str[0:2] #Keep code numerics
    data["Description"] = data["Description"].str[2:32] #Keep definition substring
    return data

def violation_code_counts():
    path = mysql_dir + "violation_code_counts.csv" #CSV file containing violation code counts
    data = pd.read_csv(path, header = None, names = ['Violation_Code','Value_Count']) # import
    data = data.sort_values(["Violation_Code","Value_Count"], ascending=[True,False]) #Sort to match code definitons
    data["Violation_Code"] = data["Violation_Code"].map("{:02}".format) #Add leading zeros
    data["Violation_Code"] = data["Violation_Code"].astype('str') #Set to string type
    data = data[data["Violation_Code"] != "00"] #ignore 00 codes which have no definition
    data = data.reset_index(drop=True) #Reset index to start from 0 
    return data

def merged_violation_codes():
    """Merge value counts and descriptions."""
    counts = violation_code_counts() #import counts
    descriptions = violation_code_descriptions() #import descriptions
    data = pd.merge(counts, descriptions, on="Violation_Code") # merge them by violation code
    return data

def value_counts_plot(save=False, N=25):
    """Plot the value counts of each violation.""" 
    codes = merged_violation_codes()
    codes = codes.sort_values("Value_Count", ascending=False)
    codes = codes.reset_index(drop=True)
    
    plt.figure(figsize=(17, 8))
    plt.title("Top 25 NYC Parking Violations (2014 - 2017)", fontsize=20)
       
    ax = plt.subplot(111)  
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
      
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()   
      
    plt.xlabel("Violation Code", fontsize=16)  
    plt.ylabel("Count", fontsize=16)  
    
    x = codes.Violation_Code.values[0:N] #Violation code
    idx = np.arange(len(x)) #Violation code placement
    ytot = codes["Value_Count"].values #All violation code counts
    ytop = ytot[0:N] #Only top N violation code counts
    percent = [i*(100.0/ytot.sum()) for i in ytot][0:N]
    
    plt.xticks(idx, x, fontsize=14)  
    plt.bar(idx, ytop, width= 0.75, color="tomato")
    plt.xlim([min(idx)-0.5,max(idx)+0.5])
    plt.tight_layout()
    
    #add percentages to bars based on total counts
    for i in range(len(percent)):
        x = i-0.40
        y = ytop[i]+1e5
        plt.text(x, y,'%.2f%%' % percent[i], fontsize=12)
    
    for i in range(len(ytop)):
        x = max(idx)-6
        y = 0.96*max(ytop)-1.75e5*i
        plt.text(x, y,
        codes.loc[i, "Violation_Code"]+ str(" ") + codes.loc[i, "Description"].strip(),
        fontsize=12)
        
    plt.text((len(idx)/2)-7, max(ytop)+1e5, "Data source: https://opendata.cityofnewyork.us | "  
             "Author: David Ascienzo (Dascienz.github.io / Dascienz@gmail.com)", fontsize=10)  
    
    if save:
        plt.savefig("violation_counts.png", format="png", dpi=300, bbox_inches="tight"); 
    plt.show()
    return

if __name__=="__main__":
    #x = readSamples(["Violation_Code"], "violations_data") #Sample 5 million rows by random.
    value_counts_plot(save=True, N=25)