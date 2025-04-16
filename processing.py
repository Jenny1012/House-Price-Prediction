import numpy as np
import pandas as pd
from math import radians,sin,cos,tan
data = pd.read_csv('C:\Users\Jenny\Desktop\house price prediction\30_Training Dataset_V2\training_data.csv')

def lonlat_to_97( lon, lat):
    """
    It transforms longitude, latitude to TWD97 system.

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees 

    Returns
    -------
    x, y [TWD97]
    """
    
    lat = radians(lat)
    lon = radians(lon)
    
    a = 6378137.0
    b = 6356752.314245
    long0 = radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1-b**2/a**2)**0.5
    e2 = e**2/(1-e**2)
    n = (a-b)/(a+b)
    nu = a/(1-(e**2)*(sin(lat)**2))**0.5
    p = lon-long0

    A = a*(1 - n + (5/4.0)*(n**2 - n**3) + (81/64.0)*(n**4  - n**5))
    B = (3*a*n/2.0)*(1 - n + (7/8.0)*(n**2 - n**3) + (55/64.0)*(n**4 - n**5))
    C = (15*a*(n**2)/16.0)*(1 - n + (3/4.0)*(n**2 - n**3))
    D = (35*a*(n**3)/48.0)*(1 - n + (11/16.0)*(n**2 - n**3))
    E = (315*a*(n**4)/51.0)*(1 - n)

    S = A*lat - B*sin(2*lat) + C*sin(4*lat) - D*sin(6*lat) + E*sin(8*lat)

    K1 = S*k0
    K2 = k0*nu*sin(2*lat)/4.0
    K3 = (k0*nu*sin(lat)*(cos(lat)**3)/24.0) * (5 - tan(lat)**2 + 9*e2*(cos(lat)**2) + 4*(e2**2)*(cos(lat)**4))

    y_97 = K1 + K2*(p**2) + K3*(p**4)

    K4 = k0*nu*cos(lat)
    K5 = (k0*nu*(cos(lat)**3)/6.0) * (1 - tan(lat)**2 + e2*(cos(lat)**2))

    x_97 = K4*p + K5*(p**3) + dx
    return x_97, y_97

university = pd.read_csv('C:\Users\Jenny\Desktop\house price prediction\30_Training Dataset_V2\external_data\大學基本資料.csv',skipinitialspace = True)
station = pd.read_csv('C:\Users\Jenny\Desktop\house price prediction\30_Training Dataset_V2\external_data\火車站點資料.csv',skipinitialspace = True)
hospital = pd.read_csv('C:\Users\Jenny\Desktop\house price prediction\30_Training Dataset_V2\external_data\醫療機構基本資料.csv',skipinitialspace = True)

print(university.info())
for i in range(len(university)):
    university.at[i,'tw97_x'], university.at[i,'tw97_y'] = lonlat_to_97(university.loc[i,'lng'], university.loc[i,'lat'])
for i in range(len(station)):
    station.at[i,'tw97_x'], station.at[i,'tw97_y'] = lonlat_to_97(station.loc[i,'lng'], station.loc[i,'lat'])
for i in range(len(hospital)):
    hospital.at[i,'tw97_x'], hospital.at[i,'tw97_y'] = lonlat_to_97(hospital.loc[i,'lng'], hospital.loc[i,'lat'])


def near(target, currentx, currenty, city):
    distance = float('inf')
    same_city = target[target['縣市名稱'] == city]
    print('here',currentx,currenty, city)
    for j in range(len(same_city)):
        dis = np.sqrt(np.power(same_city.iloc[j]['tw97_x']-currentx,2) + np.power(same_city.iloc[j]['tw97_y']-currenty,2))
        distance = min(distance, dis)
    print(distance)
    return distance

station['郵遞區號'] = station['站點地址'].str.extract('(\d+)')
station['地址'] = station['站點地址'].str.extract('(\D+)')
station['縣市名稱'] = station['地址'].str.slice(stop=3).replace(' ',"")
hospital['縣市名稱'] = hospital['縣市鄉鎮'].str.slice(stop=3).replace(' ',"")

university['縣市名稱'] = university['縣市名稱'].str.slice(start=3)
station['縣市名稱'] = station['縣市名稱'].str.replace("臺","台")
hospital['縣市名稱'] = hospital['縣市名稱'].str.replace("臺","台")
university['縣市名稱'] = university['縣市名稱'].str.replace("臺","台")

print(station['縣市名稱'])
print(hospital['縣市名稱'])
print(university['縣市名稱'])

for i in range(len(data)):    
    data.at[i,'near_university'] = near(university, data['橫坐標'].iloc[i], data['縱坐標'].iloc[i], data['縣市'].iloc[i])
    data.at[i,'near_station'] = near(station, data['橫坐標'].iloc[i], data['縱坐標'].iloc[i], data['縣市'].iloc[i])
    data.at[i,'near_hospital'] = near(hospital, data['橫坐標'].iloc[i], data['縱坐標'].iloc[i], data['縣市'].iloc[i])

import os
os.getcwd()
data.to_csv('new_training_data.csv')

