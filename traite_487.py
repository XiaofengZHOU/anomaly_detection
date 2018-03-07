#%%
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from data_helper import *
import os

file_path = 'data/data_asset_choosed/487/'
file_name = '487.csv'
df_asset = pd.read_csv(file_path+file_name)

#%%
keys = ["MDI_OBD_FUEL","ODO_FULL_METER"]

index_list = []
for i in range(len(df_asset)):
    if ' ' in df_asset.iloc[i][keys].values.tolist():
        index_list.append(i)

df_asset_drop = df_asset.drop(index_list)


#%%
df_asset_drop.to_csv(file_path+"487_drop.csv")

#%%
df_list = divise_asset_by_time(df_asset_drop,600)

for i in range( len(df_list) ) :
    df_list[i].to_csv("data/data_asset_choosed/487/subset_by_time_drop/" + str(i)+".csv",index=False)




#%%
def generate_info(df_asset,key):
    df_asset[key]=" "
    if key == "time":
        para = "recorded_at"
    if key == "distance":
        para = "ODO_FULL_METER"
    if key == "fuel":
        para = "MDI_OBD_FUEL"

    for i in range(1,len(df_asset)):
        current_data   = df_asset.iloc[i]
        last_data      = df_asset.iloc[i-1]
        current_time   = datetime_to_int(current_data["recorded_at"])
        last_time      = datetime_to_int(last_data["recorded_at"])
        if key =="time":
            current_value  = current_time
            last_value     = last_time
        else:
            current_value  = current_data[para]
            last_value     = last_data[para]

        if current_value !=" " and last_value != " ":
            df_asset.loc[i, key] = float(current_value) - float(last_value)
    return df_asset

def generate_speed_info(df_asset):
    df_asset["speed"]=" "
    for i in range(1,len(df_asset)):
        current_data   = df_asset.iloc[i]
        distance = current_data["distance"]
        time = current_data["time"]
        speed = distance/time*3.6
        df_asset.loc[i, "speed"] = speed
    return df_asset


def generate_average_speed(df_asset,time_interval):
    key_name = "average_speed_"+str(time_interval)
    df_asset[key_name]=" "
    for i in range(len(df_asset)-1):
        if df_asset.loc[i,key_name] != " " and df_asset.loc[i+1,key_name] != " ":
            continue

        current_data   = df_asset.iloc[i]
        current_meter  = current_data["ODO_FULL_METER"]
        current_time   = datetime_to_int(current_data["recorded_at"])

        next_valide_time  = None
        next_valide_meter = None
        for k in range(i+1,len(df_asset)):
            next_valide_time  = datetime_to_int(df_asset.iloc[k]['recorded_at'])
            if next_valide_time -  current_time >=60:
                next_valide_meter = df_asset.iloc[k]["ODO_FULL_METER"]
                break
        if next_valide_time != None and next_valide_meter !=None:
            interval_time  = next_valide_time - current_time
            interval_value = float(next_valide_meter) - float(current_meter)
            for j in range(i+1,k+1):
                df_asset.loc[j,key_name] = str(interval_value/interval_time*3.6)
    return df_asset


def fill_blank_average_speed(df_asset,key_name):
    for i in range(1,len(df_asset)):
        if df_asset.loc[i,key_name] == " ":
            if i == len(df_asset)-1:
                df_asset.loc[i,key_name] = df_asset.loc[i,"speed"]
            else:

                previous_data   = df_asset.iloc[i-1]
                previous_meter  = previous_data["ODO_FULL_METER"]
                previous_time   = datetime_to_int(previous_data["recorded_at"])

                last_data = df_asset.iloc[-1]
                last_meter  = last_data["ODO_FULL_METER"]
                last_time   = datetime_to_int(last_data["recorded_at"])

                interval_time  = float(last_time) - float(previous_time)
                interval_value = float(last_meter) - float(previous_meter)
                for j in range(i,len(df_asset)):
                    df_asset.loc[j,key_name] = str(interval_value/interval_time*3.6)
                break
    return df_asset

def merge_data_by_average_speed(df_asset,key_name):
    drop_list = []
    for i in range(len(df_asset)):
        current_data   = df_asset.iloc[i]
        current_speed  = current_data[key_name]

        for k in range(i+1,len(df_asset)):
            if df_asset.iloc[k][key_name] == current_speed:
                drop_list.append(i)

    df_asset = df_asset.drop(drop_list)
    return df_asset

#%%

file_path_out = 'data/data_asset_choosed/487/subset_by_time_drop_merge/'
file_path = 'data/data_asset_choosed/487/subset_by_time_drop/'
for file_name in os.listdir(file_path):
    df_i = pd.read_csv(file_path+file_name)
    df_i = generate_info(df_i,"time")
    df_i = generate_info(df_i,"distance")
    df_i = generate_speed_info(df_i)
    df_i = generate_average_speed(df_i,50)
    df_i = fill_blank_average_speed(df_i,"average_speed_50")
    df_i = generate_info(df_i,"fuel")
    df_i.to_csv(file_path_out + file_name,index=False)


#%%

file_path_out = 'data/data_asset_choosed/487/subset_by_time_drop_merge/'
file_path = 'data/data_asset_choosed/487/subset_by_time_drop/'
for file_name in os.listdir(file_path):
    df_i = pd.read_csv(file_path+file_name)
    df_i = merge_data_by_average_speed(df_i,"average_speed_50")
    df_i.to_csv(file_path_out + file_name,index=False)


#%%
file_path = 'data/data_asset_choosed/487/subset_by_time_drop_merge/'
for file_name in os.listdir(file_path):
    df_i = pd.read_csv(file_path+file_name)
    df_i = generate_info(df_i,"time")
    df_i = generate_info(df_i,"distance")
    df_i = generate_info(df_i,"fuel")
    df_i.to_csv(file_path + file_name,index=False)
