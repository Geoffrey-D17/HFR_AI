# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:43:05 2025

@author: geofd
"""

from Loaddata.Def_et_biblio import *
from Loaddata.PAB_data import *
from Loaddata.PAO_data import *

liste_1 = np.arange(0, len(points_lon_lat),1)
l = pd.DataFrame(liste_1, columns = ['nb_point'])

# peaks_tab2, val_peak2, peaks_freq2, R2, time_pxy, time_ur, u_r = tretement_data(
#     PXY_files, Ur_files, freq, coordonnes_lon_lat, points_lon_lat, liste_1, l)

peaks_tab2 = {}
val_peak2 = {}
peaks_freq2 = {}
R2 = []
time_pxy = []
time_ur = []
u_r = []

for i, pxy_file in enumerate(PXY_files):
    
    time_str_pxy = extract_time_from_filename(pxy_file)
    time_pxy.append(time_str_pxy)
    try:
        Pxy = loadmat(f"D:/Data_HFR/PAB/SPEC_mat/{pxy_file}", mat_dtype=True)
        # Pxy = loadmat(f"E:/HFR/WERA/PAO/SPEC/{pxy_file}")  # Charger PXY (mat)
        Pxy_2 = np.array(Pxy['PXY'], dtype=object)
        valid_indices = (points_lon_lat.lon_spec < Pxy_2.shape[0]) & (points_lon_lat.lat_spec < Pxy_2.shape[1])
    
        filtered_lon_spec = points_lon_lat.lon_spec[valid_indices]
        filtered_lat_spec = points_lon_lat.lat_spec[valid_indices]
        filtered_lon_spec_coo = coordonnes_lon_lat.lon_spec[valid_indices]
        filtered_lat_spec_coo = coordonnes_lon_lat.lat_spec[valid_indices]
        
        spectre = Pxy_2[filtered_lon_spec, filtered_lat_spec]
        peaks_tab, val_peak, peaks_freq, R = detect_peaks_and_calculate(spectre, freq, time_str_pxy, filtered_lon_spec, filtered_lat_spec)
        peaks_freq2_df = pd.DataFrame(peaks_freq).T
        
        peaks_tab2[i] = pd.DataFrame(peaks_freq).T
        val_peak2[i] = pd.DataFrame(val_peak).T
        peaks_freq2[i] = pd.DataFrame(peaks_freq).T 
        R2.append(R)
    except:
        print("An exception occurred")

for ur_file in Ur_files:
    time_str_ur = extract_time_from_filename(ur_file)
    time_ur.append(time_str_ur)
    # crad_u = pd.read_csv(f"C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Radar HF/WERA/PAO/crad_/{ur_file}", header=None)
    crad_ = loadmat(f"D:/Data_HFR/PAB/CRAD_mat2/{ur_file}", mat_dtype=True)
    crad_u = np.array(crad_['Ur'], dtype=object)
    crad_lon = np.array(crad_['Y'], dtype=object)
    crad_lat = np.array(crad_['X'], dtype=object)
    # crad_u = crad_u.to_numpy()
    # print('oui')
    
    valid_points = (points_lon_lat.lon_crad < crad_u.shape[0]) & (points_lon_lat.lat_crad < crad_u.shape[1])
    filtered_lon = points_lon_lat.lon_crad[valid_points]
    filtered_lat = points_lon_lat.lat_crad[valid_points]
    
    # Extraire les valeurs de crad_u pour ces indices
    u_radial = pd.DataFrame(crad_u[filtered_lon, filtered_lat], columns=['Ur'])
    time_repeated = np.repeat(time_str_ur, len(points_lon_lat))
    t = pd.DataFrame(time_repeated, columns = ['time'])
    u_r.append(pd.concat([u_radial['Ur'], l['nb_point'], t['time'], 
                          coordonnes_lon_lat.lon_crad[valid_points], coordonnes_lon_lat.lat_crad[valid_points],
                          points_lon_lat.lon_spec, points_lon_lat.lat_spec], axis = 1))

peaks_freq3 = pd.concat(peaks_freq2, axis = 0) 
val_peak3 = pd.concat(val_peak2, axis = 0)
peaks_tab3 = pd.concat(peaks_tab2, axis = 0)
peaks_tab3
u_r3 = pd.concat(u_r, axis = 0)
peaks_tab3 = peaks_tab3.rename(columns={0:'peak_m', 1:'peak_p', 2: 'time', 3: 'point', 4:'lon_p', 5:'lat_p'})
val_peak3 = val_peak3.rename(columns={0:'vpeak_m', 1:'vpeak_p', 2: 'time', 3: 'point', 4:'lon_p', 5:'lat_p'})
peaks_freq3 = peaks_freq3.rename(columns={0:'fpeak_m', 1:'fpeak_p', 2: 'time', 3: 'point', 4:'lon_p', 5:'lat_p'})
data_spec = pd.concat([peaks_freq3[['fpeak_m', 'fpeak_p', 'time']], 
                       val_peak3[['vpeak_m','vpeak_p']], 
                       peaks_tab3[['peak_m','peak_p', 'time' ,'point','lon_p','lat_p']]], axis = 1)
f_B = 0.4102
data_spec['offset_m'] = data_spec['fpeak_m'] + f_B
data_spec['offset_p'] = data_spec['fpeak_p'] - f_B

dict_data = {'spec_data' : data_spec,
             'crad_data' : u_r3,
             'points_lon_lat' : points_lon_lat,
             'coordonnes_lon_lat' : coordonnes_lon_lat,
             'time_ur' : time_ur,
             'time_pxy' : time_pxy}

import pickle 
# Sauvegarde du dictionnaire
with open('D:/Data_HFR/PAB/dict_data_PAB_2020_v2.pkl', 'wb') as f:
    pickle.dump(dict_data, f)
