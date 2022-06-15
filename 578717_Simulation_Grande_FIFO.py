#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:20:01 2022

@author: orkuntunay
"""
#%% Import packages
import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
from random import seed
from random import gauss
from pyproj import Geod
import time
#%% Functions for Simulation

# Hourly demand generator
def demand_generator(demand_hour):
    demand_hour['Simulation'] = demand_hour.apply(
        lambda row: int(round(gauss(row.Mean_Demand, row.Std))), axis=1)
    demand_hour["Simulation"][demand_hour['Simulation'] <0] = 0
    return demand_hour[["Requested Weekday", "Requested Hour", "Mean_Demand","Simulation"]]

# Haversine distance 
def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km

# Get the closest vehicle to the incident
def closest_v(towing_v, towing_v_loc, acc_lon, acc_lat):
    dist_raw = [haversine_vectorize(i[0],i[1], acc_lon, acc_lat)\
                for i in towing_v_loc]
    dist = np.multiply(dist_raw,towing_v)
    dist[np.where(towing_v == 0)] = 1000000
    res = np.where(dist == np.min(dist))[0][0]
    return res

# Initialize variables for the simulation
def initialize(no_vehicles, initial_loc, service_time, avg_speed):
    no_towing_v = no_vehicles
    towing_v = np.ones(no_towing_v)
    towing_v_loc = [initial_loc for v in range(0,no_towing_v)]
    towing_v_loc = np.array(towing_v_loc)

    return towing_v, towing_v_loc, service_time, avg_speed

def event_generator(data_n, demand_hour):

    sample = data_n[(data_n["Within Citi Center (20KM)"]== True)]
    sample["Day"] = sample["Requested Date"].dt.weekday
    sample["Hour"] = sample["Requested Date"].dt.hour
    sample["Minute"] = sample["Requested Date"].dt.minute
    sample["Drop-off Type"] = 1
    sample.loc[(sample["Drop-off Latitude"].isna())&(
        sample["Drop-off Longitude"].isna()),"Drop-off Type"] = 0
    
    tmp = pd.DataFrame(columns=sample.columns)
    
    #Generate Hourly Demamd Based on 
    demand_hourly = demand_generator(demand_hour)
    
    for i in range(0,7):
        for j in range(0,24):    
            d = demand_hourly[(demand_hourly["Requested Weekday"]==i)&\
                              (demand_hourly["Requested Hour"]==j)]["Simulation"].values[0]
            tmp = tmp.append(sample[(sample["Day"]==i)&\
                                    (sample["Hour"]==j)].sample(d).sort_values(
                                        by="Minute"))
              
                        
    #tmp1 = tmp.head(50)
    tmp1 = tmp.copy()
    
    event = tmp1[["Claim No","Requested Date","Day","Hour","Minute",
                  "Pick-up Longitude","Pick-up Latitude",
                  "Drop-off Type","Drop-off Longitude","Drop-off Latitude"]]
    
    event["Requested Date"] = dt.datetime(2022, 8, 1)\
        +pd.to_timedelta(event['Day'], unit='d')\
            +pd.to_timedelta(event['Hour'], unit='h')\
                +pd.to_timedelta(event['Minute'], unit='m')
    
    
    event.rename(columns = {"Claim No" : "Info",
                            'Pick-up Longitude':'Longitude',
                            'Pick-up Latitude':'Latitude'}, inplace = True)
    event["Assignment Time"] = None
    event["Type"] = "Accident"
    event["Status"] = 1
    event["Completed Time"] = 0
    event["Serviced Time"] = 0
    event["Servicing Vehicle"] = None
    event["Availability Log"] = None
    event["Location Log"] = None
    event["Queue Length"] = None
    event["Queue Log"] = None
    event = event.reset_index()
    event = event.drop(columns = ['index'])

    return event

def move_to_city(towing_v, towing_v_loc, time_min, avg_speed,city_center_loc):   
    g = Geod(ellps='clrk66')
    #Maximum distance that can be travelled in the time between two events
    dist_max = (avg_speed * (time_min/60))*1000    
    for i in range(0,len(towing_v)):
        #Move the vehicle if the vehicle is available within the period
        if towing_v[i] == 1:
            
            az12,az21,dist = g.inv(towing_v_loc[i][0],towing_v_loc[i][1],
                                   city_center_loc[0],city_center_loc[1])
            
            #If the distance to the city center is higher than the maximum
            #distance that can be travelled
            if dist > dist_max:
                endlon, endlat, backaz = g.fwd(towing_v_loc[i][0],towing_v_loc[i][1],
                                               az12, dist_max)
                towing_v_loc[i] = [endlon,
                                   endlat]
            else:
                towing_v_loc[i] = [city_center_loc[0],
                                   city_center_loc[1]]
    return towing_v_loc

#%% Load Data
wd = "/Users/orkuntunay/Desktop/RSM/1.0 Thesis/2-Python"
os.chdir(wd)
data = pd.read_csv(os.getcwd() + '/RLTowingAndFix_v2.csv')

#%% Date Time Format
data['Incident Date'] = pd.to_datetime(data['Incident Date'],format='%d.%m.%Y %H:%M')
data['Requested Date'] = pd.to_datetime(data['Requested Date'],format='%d.%m.%Y %H:%M')
data['Promised Arrival Date'] = pd.to_datetime(data['Promised Arrival Date'],format='%d.%m.%Y %H:%M')
data['Assignement Date'] = pd.to_datetime(data['Assignement Date'],format='%d.%m.%Y %H:%M')
data['Arrival Date'] = pd.to_datetime(data['Arrival Date'],format='%d.%m.%Y %H:%M')
data['Completed Date'] = pd.to_datetime(data['Completed Date'],format='%d.%m.%Y %H:%M')

#%% Filter Nicosia
data_n = data[(data["Pick-up Municipality"] == "NICOSIA") &\
              (data["Drop-off Location"].str.split(",").str[-1] == " Lefkosia")]
#Remove duplicate accident codes    
data_n = data_n.drop_duplicates(subset='Claim No', keep="last")    

#%% Create An Hourly Demand Dataset
tmp = data_n.copy()
tmp["Requested Day"] = tmp["Requested Date"].dt.date
tmp["Requested Hour"] = tmp["Requested Date"].dt.hour
tmp = tmp[["Requested Day", "Requested Hour", "Claim No"]].groupby(
    by=["Requested Day", "Requested Hour"],as_index=False).count()

all_days = np.unique(tmp["Requested Day"])
all_hours = np.unique(tmp["Requested Hour"])
names = ["Requested Day", "Requested Hour"]
mind = pd.MultiIndex.from_product(
    [all_days, all_hours], names=names)

demand = tmp.set_index(names).reindex(mind, fill_value=0).reset_index()
demand["Requested Weekday"] = demand["Requested Day"].dt.weekday
demand["Requested Day"] = demand["Requested Day"] + pd.to_timedelta(
    demand["Requested Hour"], unit='h')

# Hourly Historical Demand

demand_hour = demand.groupby(by = ["Requested Hour", "Requested Weekday"])["Claim No"].sum().reset_index()
demand_hour["Total_Days"] = demand.groupby(by = ["Requested Hour", "Requested Weekday"])["Claim No"].count().reset_index()["Claim No"]
demand_hour["Std"] = demand.groupby(by = ["Requested Hour", "Requested Weekday"])["Claim No"].std().reset_index()["Claim No"]
demand_hour["Mean_Demand"] = demand_hour["Claim No"]/demand_hour["Total_Days"] 


#%%
#CALL EVENT GENERATOR AND CREATE A LIST OF EVENTS HAVING 1000 SAMPLE SETS
event_master = pd.DataFrame()
for s in range(0,1000):
    event_tmp = event_generator(data_n, demand_hour)
    even_tmp = event_tmp.drop_duplicates()
    even_tmp["Sim_No"] = s
    event_master = event_master.append(even_tmp)
#     print(s)
event_master.to_csv("event_master.csv")
# ##
#%%
city_center_loc = [data_n["Pick-up Longitude"].mean(),
                   data_n["Pick-up Latitude"].mean()]
no_simulations = 1000
#event_master = pd.read_csv("event_master.csv", index_col = False).iloc[:,1:]

result = pd.DataFrame()
for v in range(3,8):

    event_full = pd.DataFrame()
    for s in range(0,no_simulations):
        no_towing_v = v
        towing_v, towing_v_loc, service_time, avg_speed = initialize(no_towing_v,
                                                                     city_center_loc,
                                                                     10,
                                                                     30)
        # event = event_generator(data_n, demand_hour)
        # event = event.drop_duplicates()
        # event_acc = event.copy()
        event = event_master[event_master["Sim_No"]==s].drop(
            columns="Sim_No").reset_index().drop(columns = "index")
        event["Requested Date"] = pd.to_datetime(event["Requested Date"])
        event["Assignment Time"] = None
        event["Type"] = "Accident"
        event["Status"] = 1
        event["Completed Time"] = 0
        event["Serviced Time"] = 0
        event["Servicing Vehicle"] = None
        event["Availability Log"] = None
        event["Location Log"] = None
        event["Queue Length"] = None
        event["Queue Log"] = None
        event_acc = event.copy()
        
        queue = []
        queue_log = []
        
        i = 0
        while np.sum(event["Status"])>0:
            
            if i!=0:
                time_min = (event.iloc[i,:]["Requested Date"]-event.iloc[i-1,:][
                    "Requested Date"]).total_seconds()/60.0
                towing_v_loc = move_to_city(towing_v, towing_v_loc, time_min, 
                                            avg_speed,city_center_loc)
            
    
            #Check the Event Type / Type: Accident
            if event.at[i,"Type"] == "Accident":            
                # Keep logs of vehicle just before the event
                event.at[i,"Availability Log"] = list(towing_v)
                event.at[i,"Location Log"] = towing_v_loc.tolist()
                
                #Check Available Towing Vehicles
                #If No Vehicle Available Wait in the Queue
                if np.sum(towing_v) == 0:
                    queue.append(event.at[i,"Info"])
                    queue_log.append(event.at[i,"Info"])
                    event.at[i,"Queue Length"] = len(queue)
                    event.at[i,"Queue Log"] = np.array(queue).tolist()
                
                #If A Vehicle is Available Send to the Accident Location
                elif np.sum(towing_v) > 0:
                    
                    #Log Queue
                    event.at[i,"Queue Length"] = len(queue)
                    event.at[i,"Queue Log"] = np.array(queue).tolist()    
                                
                    #Get the coordinates of the accident
                    acc_lon = event.iloc[i,:]["Longitude"]
                    acc_lat =  event.iloc[i,:]["Latitude"]
                    
                    #Get the closest vehicle to the accident
                    tow_i = closest_v(towing_v, towing_v_loc, acc_lon,acc_lat)
                    
                    #If no accidents in the queue, assign immediately
                    event.at[i,"Assignment Time"] = event.at[i,"Requested Date"]
                    
                    #Calculate Distance of the Vehicle to the Location
                    distance = haversine_vectorize(towing_v_loc[tow_i][0],
                                                   towing_v_loc[tow_i][1],
                                                   acc_lon,
                                                   acc_lat)
                    
                    if event.at[i,"Drop-off Type"] == 1:
                        do_lon = event.iloc[i,:]["Drop-off Longitude"]
                        do_lat =  event.iloc[i,:]["Drop-off Latitude"]                
                        do_distance = haversine_vectorize(acc_lon,
                                                          acc_lat,
                                                          do_lon,
                                                          do_lat)
                        do_travel_time = round(do_distance/(avg_speed/60))
                        #Set Vehicle Location
                        towing_v_loc[tow_i] = [do_lon,
                                               do_lat]
                    else:
                        do_travel_time = 0
                        #Set Vehicle Location
                        towing_v_loc[tow_i] = [acc_lon,
                                               acc_lat]
                    
                    #Calculate Travel Time
                    travel_time = round(distance/(avg_speed/60))
                    event.at[i,"Completed Time"] = (event.at[i,"Requested Date"] \
                        + dt.timedelta(
                            minutes = travel_time + service_time + do_travel_time))            
                    
                    #Calculate Serviced Time    
                    event.at[i,"Serviced Time"] = (event.at[i,"Requested Date"] \
                        + dt.timedelta(
                            minutes = travel_time + service_time))
                    
                    #Record the Servicing Vehicle
                    event.at[i,"Servicing Vehicle"] = "v"+str(tow_i)               
                        
                    #Set Vehicle Unavailable                
                    towing_v[tow_i]=0
                    
                    #Define Vehicle Available Event
                    event_va = ["v"+str(tow_i),
                                event.at[i,"Completed Time"],
                                event.at[i,"Completed Time"].weekday(),
                                event.at[i,"Completed Time"].time().hour,
                                event.at[i,"Completed Time"].time().minute,
                                towing_v_loc[tow_i][0],
                                towing_v_loc[tow_i][1],
                                None, None, None, None,
                                "Vehicle Available",
                                1,
                                None,None,None,None,None,None,None]
                    
                    tmp2 = pd.DataFrame(np.array(event_va).reshape(
                        -1,len(event_va)),columns=event.columns)
                    
                    #Events before vehicle becomes available
                    tmp1 = event[event["Requested Date"]-event.at[i,"Completed Time"]<=\
                          pd.Timedelta(0)]
                    
                    #Events after vehicle becomes available
                    tmp3 = event[event["Requested Date"]-event.at[i,"Completed Time"]>\
                          pd.Timedelta(0)]
                    
                    #Insert Vehicle Available Event
                    event = pd.concat([tmp1,tmp2,tmp3],ignore_index = True)
                    event.at[i,"Status"] = 0
                    
                    #print(str(i) + " " + str(event.at[i,"Completed Time"]))
        
            #If A Vehicle Becomes Available
            if event.at[i,"Type"] == "Vehicle Available":
                # Keep logs of vehicle just before the event
                event.at[i,"Availability Log"] = list(towing_v)
                event.at[i,"Location Log"] = towing_v_loc.tolist()
                
                #Log Queue
                event.at[i,"Queue Length"] = len(queue)
                event.at[i,"Queue Log"] = np.array(queue).tolist()    
                
                # Set the vehicle available
                towing_v[int(event.at[i,"Info"][1:])] = 1
                #Toggle Event Status
                event.at[i,"Status"] = 0
                # Check if there are any accidents in the queue        
                # The Dispatching Algorithm comes into play
                if len(queue) > 0:
                    
                    #FIFO, process the accident that happened first
                    acc = queue.pop(0)
                    
                    #Get the index of the accident
                    i_tmp = event[event["Info"]==acc].index[0]            
                    
                    #Get the coordinates of the accident
                    acc_lon = event.iloc[i_tmp,:]["Longitude"]
                    acc_lat =  event.iloc[i_tmp,:]["Latitude"]
                    
                    #Get the index of the closest available vehicle to the accident
                    tow_i = closest_v(towing_v, towing_v_loc, acc_lon,acc_lat)
                    
                    
                    #Assignment time is the time when the vehicle becomes available 
                    event.at[i_tmp,"Assignment Time"] = event.at[i,"Requested Date"]
                    
                    #Calculate Distance of the Vehicle to the Location
                    distance = haversine_vectorize(towing_v_loc[tow_i][0],
                                                   towing_v_loc[tow_i][1],
                                                   acc_lon,
                                                   acc_lat)
        
                    if event.at[i_tmp,"Drop-off Type"] == 1:
                        do_lon = event.iloc[i_tmp,:]["Drop-off Longitude"]
                        do_lat =  event.iloc[i_tmp,:]["Drop-off Latitude"]                
                        do_distance = haversine_vectorize(acc_lon,
                                                          acc_lat,
                                                          do_lon,
                                                          do_lat)
                        do_travel_time = round(do_distance/(avg_speed/60))
                        #Set Vehicle Location
                        towing_v_loc[tow_i] = [do_lon,
                                               do_lat]
                    else:
                        do_travel_time = 0
                        #Set Vehicle Location
                        towing_v_loc[tow_i] = [acc_lon,
                                               acc_lat]
        
                    #Calculate Travel Time
                    travel_time = round(distance/(avg_speed/60))
                    event.at[i_tmp,"Completed Time"] = (event.at[i,"Requested Date"]\
                        + dt.timedelta(minutes = travel_time + service_time + do_travel_time))
                    
                    #Calculate Service Time
                    event.at[i_tmp,"Serviced Time"] = (event.at[i,"Requested Date"]\
                        + dt.timedelta(minutes = travel_time + service_time))                
                        
                    #Set Vehicle Location
                    towing_v_loc[tow_i] = [event.iloc[i_tmp,:]["Longitude"],
                                            event.iloc[i_tmp,:]["Latitude"]]
                    
                    #Set Vehicle Unavailable                
                    towing_v[tow_i]=0
                    
                    #Record the Servicing Vehicle
                    event.at[i_tmp,"Servicing Vehicle"] = "v"+str(tow_i)   
        
                    #Define Vehicle Available Event
                    event_va = ["v"+str(tow_i),
                                event.at[i_tmp,"Completed Time"],
                                event.at[i_tmp,"Completed Time"].weekday(),
                                event.at[i_tmp,"Completed Time"].time().hour,
                                event.at[i_tmp,"Completed Time"].time().minute,
                                towing_v_loc[tow_i][0],
                                towing_v_loc[tow_i][1],
                                None, None, None, None,
                                "Vehicle Available",
                                1,
                                None,None, None,None,None,None,None]
        
                    tmp2 = pd.DataFrame(np.array(event_va).reshape(
                        -1,len(event_va)),columns=event.columns)
                    
                    #Events before vehicle becomes available
                    tmp1 = event[event["Requested Date"]-event.at[i_tmp,"Completed Time"]<=\
                          pd.Timedelta(0)]
                    
                    #Events after vehicle becomes available
                    tmp3 = event[event["Requested Date"]-event.at[i_tmp,"Completed Time"]>\
                          pd.Timedelta(0)]
                    
                    #Insert Vehicle Available Event
                    event = pd.concat([tmp1,tmp2,tmp3],ignore_index = True)    
                    
                    #Toggle Event Status as the accident is served
                    event.at[i_tmp,"Status"] = 0
                
                  
            i += 1
        print(str(v) + "," + str(s))  
        event["Simulation"] = s
        event_full = event_full.append(event)
    event_full["Number of Vehicles"] = v   
    result = result.append(event_full)
    
result["Response Time"] = (pd.to_datetime(result["Serviced Time"])\
                                 -result["Requested Date"])/ pd.Timedelta(minutes=1)

result.to_csv("Result_1000sim_Tow_3to7.csv")
    
#%% Analysis of the Results

result = pd.read_csv("Result_1000sim_Tow_4to7.csv")

tmp1 = result[(result["Number of Vehicles"]==4) &(
    result["Day"]==1)&(result["Hour"]==2)&(result["Type"]=="Accident")]

tmp = result[["Number of Vehicles","Day","Hour","Response Time"]].groupby([
    "Number of Vehicles","Day","Hour"]).mean()


fig = px.box(result, x="Number of Vehicles", y="Response Time", color = "Number of Vehicles")
fig.show()
fig.show(renderer="browser")


#%%Histogram
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Number of Vehicles: 4","Number of Vehicles: 5",
                                    "Number of Vehicles: 6","Number of Vehicles: 7"),
                    shared_yaxes = 'all',
                    shared_xaxes = 'all')

fig.add_trace(go.Histogram( 
                   x=result[result["Number of Vehicles"]==4]["Response Time"],
                   nbinsx = 20),
                 row=1, col=1)
fig.add_trace(go.Histogram( 
                   x=result[result["Number of Vehicles"]==5]["Response Time"],
                   nbinsx = 20),
                 row=1, col=2)
fig.add_trace(go.Histogram( 
                   x=result[result["Number of Vehicles"]==6]["Response Time"],
                   nbinsx = 20),
                 row=2, col=1)
fig.add_trace(go.Histogram( 
                   x=result[result["Number of Vehicles"]==7]["Response Time"],
                   nbinsx = 20),
                 row=2, col=2)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Vehicle Availability Calculation
result["Vehicle Availability"] = None
for i in range(0,len(result)):
    tmp_lst = result.loc[i,"Availability Log"].strip('][').split(', ')
    tmp_lst = [float(i) for i in tmp_lst]
    ppt = (sum(tmp_lst)/len(tmp_lst))
    result.at[i,"Vehicle Availability"] = ppt

va = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Vehicle Availability"]].groupby(
        "Number of Vehicles").apply(np.mean)["Vehicle Availability"].values

rt = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Response Time"]].groupby(
        "Number of Vehicles").apply(np.mean)["Response Time"].values       
        
# Create traces
fig = make_subplots(rows=2, cols=1, shared_yaxes=True)

fig.add_trace(go.Scatter(x=[4,5,6,7], y= va),
              row=1, col=1)

fig.add_trace(go.Scatter(x=[4,5,6,7], y=rt),
              row=2, col=1)
fig.update_layout(showlegend=False)
fig.update_layout(title="Vehicle Availability Ratio and the Response Time by Number of Vehicles")
fig.update_xaxes(type='category')
fig.show()
fig.show(renderer="browser")

#%% Accidents served by vehicle

tmp = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Servicing Vehicle","Info"]].groupby([
        "Number of Vehicles","Servicing Vehicle"]).count().reset_index()

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Number of Vehicles: 4","Number of Vehicles: 5",
                                    "Number of Vehicles: 6","Number of Vehicles: 7"))

fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==4]["Servicing Vehicle"],
                   y=tmp[tmp["Number of Vehicles"]==4]["Info"]),
                 row=1, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==5]["Servicing Vehicle"],
                   y=tmp[tmp["Number of Vehicles"]==5]["Info"]),
                 row=1, col=2)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==6]["Servicing Vehicle"],
                   y=tmp[tmp["Number of Vehicles"]==6]["Info"]),
                 row=2, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==7]["Servicing Vehicle"],
                   y=tmp[tmp["Number of Vehicles"]==7]["Info"]),
                 row=2, col=2)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Queuing Results by Hour
result["Queue"]=0
result.loc[(result["Assignment Time"]!=result["Requested Date"])&(
    result["Type"]== "Accident"),"Queue"]=1


tmp = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Hour","Queue"]].groupby([
        "Number of Vehicles","Hour"]).sum().reset_index()

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Number of Vehicles: 4","Number of Vehicles: 5",
                                    "Number of Vehicles: 6","Number of Vehicles: 7"),
                    shared_yaxes = 'all',
                    shared_xaxes = 'all')

fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==4]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==4]["Queue"]),
                 row=1, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==5]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==5]["Queue"]),
                 row=1, col=2)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==6]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==6]["Queue"]),
                 row=2, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==7]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==7]["Queue"]),
                 row=2, col=2)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Queuing Results by Day
result["Queue"]=0
result.loc[(result["Assignment Time"]!=result["Requested Date"])&(
    result["Type"]== "Accident"),"Queue"]=1


tmp = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Day","Queue"]].groupby([
        "Number of Vehicles","Day"]).sum().reset_index()

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Number of Vehicles: 4","Number of Vehicles: 5",
                                    "Number of Vehicles: 6","Number of Vehicles: 7"),
                    shared_yaxes = 'all',
                    shared_xaxes = 'all')

fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==4]["Day"],
                   y=tmp[tmp["Number of Vehicles"]==4]["Queue"]),
                 row=1, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==5]["Day"],
                   y=tmp[tmp["Number of Vehicles"]==5]["Queue"]),
                 row=1, col=2)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==6]["Day"],
                   y=tmp[tmp["Number of Vehicles"]==6]["Queue"]),
                 row=2, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==7]["Day"],
                   y=tmp[tmp["Number of Vehicles"]==7]["Queue"]),
                 row=2, col=2)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Mean Response Time by Hour

tmp = result[result["Type"]=="Accident"][[
    "Number of Vehicles","Hour","Response Time"]].groupby([
        "Number of Vehicles","Hour"]).mean().reset_index()

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Number of Vehicles: 4","Number of Vehicles: 5",
                                    "Number of Vehicles: 6","Number of Vehicles: 7"),
                    shared_yaxes = 'all',
                    shared_xaxes = 'all')

fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==4]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==4]["Response Time"]),
                 row=1, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==5]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==5]["Response Time"]),
                 row=1, col=2)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==6]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==6]["Response Time"]),
                 row=2, col=1)
fig.add_trace(go.Bar( 
                   x=tmp[tmp["Number of Vehicles"]==7]["Hour"],
                   y=tmp[tmp["Number of Vehicles"]==7]["Response Time"]),
                 row=2, col=2)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Geographic Heatmap of Incidents
from folium import Map
from folium.plugins import HeatMap
import webbrowser
tmp = result[result["Hour"]==1]
tmp = tmp[tmp["Type"]=="Accident"]
for_map = Map(location=[np.mean(tmp["Latitude"]), 
                        np.mean(tmp["Longitude"])], zoom_start=11, )

hm_wide = HeatMap(
    list(zip(tmp["Latitude"].values, 
             tmp["Longitude"].values)),
    min_opacity=0.2,
    radius=17, 
    blur=15, 
    max_zoom=1,
)

for_map.add_child(hm_wide)
for_map.save("Incident_Heatmap.html")

chrome_path = 'open -a /Applications/Google\ Chrome.app %s'

webbrowser.get(chrome_path).open("Incident_Heatmap.html")

#%% Distributions of the Simulated Demand by Weekday

fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("Historical Demand Average","","Number of Vehicles: 4",
                                    "Number of Vehicles: 5","Number of Vehicles: 6",
                                    "Number of Vehicles: 7"))

fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==4)&(result["Type"]=="Accident")]["Day"],
                   nbinsx = 20),
                 row=2, col=1)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==5)&(result["Type"]=="Accident")]["Day"],
                   nbinsx = 20),
                 row=2, col=2)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==6)&(result["Type"]=="Accident")]["Day"],
                   nbinsx = 20),
                 row=3, col=1)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==7)&(result["Type"]=="Accident")]["Day"],
                   nbinsx = 20),
                 row=3, col=2)
fig.add_trace(go.Bar( 
                   x=demand_hour[["Requested Weekday","Mean_Demand"]].groupby("Requested Weekday").sum().reset_index()["Requested Weekday"],
                   y=demand_hour[["Requested Weekday","Mean_Demand"]].groupby("Requested Weekday").sum().reset_index()["Mean_Demand"]),
                 row=1, col=1)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")

#%% Distributions of the Simulated Demand by Hour

fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("Historical Demand Average","","Number of Vehicles: 4",
                                    "Number of Vehicles: 5","Number of Vehicles: 6",
                                    "Number of Vehicles: 7"))

fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==4)&(result["Type"]=="Accident")]["Hour"],
                   nbinsx = 24),
                 row=2, col=1)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==5)&(result["Type"]=="Accident")]["Hour"],
                   nbinsx = 24),
                 row=2, col=2)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==6)&(result["Type"]=="Accident")]["Hour"],
                   nbinsx = 24),
                 row=3, col=1)
fig.add_trace(go.Histogram( 
                   x=result[(result["Number of Vehicles"]==7)&(result["Type"]=="Accident")]["Hour"],
                   nbinsx = 24),
                 row=3, col=2)
fig.add_trace(go.Bar( 
                   x=demand_hour[["Requested Hour","Mean_Demand"]].groupby("Requested Hour").sum().reset_index()["Requested Hour"],
                   y=demand_hour[["Requested Hour","Mean_Demand"]].groupby("Requested Hour").sum().reset_index()["Mean_Demand"]),
                 row=1, col=1)
fig.update_layout(showlegend=False)
fig.show()
fig.show(renderer="browser")
