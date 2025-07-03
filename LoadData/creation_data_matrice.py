# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:57:53 2025

@author: geofd
"""

from Loaddata.Def_et_biblio import *
from Loaddata.creation_data_liste import *

#%% creation de matrices des données

df = peaks_tab2[0]
# all_values = np.concatenate([df.values for df in peaks_tab2.values()])
# Trouver les dimensions max en lon et lat
# max_lon = max(df[3].max() for df in peaks_tab2.values()) + 1
# max_lat = max(df[4].max() for df in peaks_tab2.values()) + 1
nb_temps = len(peaks_tab2)

max_lon = max(df['lon_spec'].max() for df in u_r) + 1
max_lat = max(df['lat_spec'].max() for df in u_r) + 1
nb_temps = len(u_r)

matrice_ur = np.full((nb_temps, max_lon, max_lat), np.nan)

for t, df in enumerate(u_r):
    for _, row in df.iterrows():
        lon_idx = int(row['lon_spec'])
        lat_idx = int(row['lat_spec'])
        matrice_ur[t, lon_idx, lat_idx] = row['Ur'] 
matrice_ur2 = matrice_ur.T


# Initialiser la matrice 3D avec NaN
matrice_peaks_tab_n = np.full((nb_temps, max_lon, max_lat), np.nan)
matrice_peaks_tab_p = np.full((nb_temps, max_lon, max_lat), np.nan)

# Remplissage de la matrice
for t, df in peaks_tab2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])  # Indice de longitude
        lat_idx = int(row[5])  # Indice de latitude
        matrice_peaks_tab_n[t, lon_idx, lat_idx] = row[0]  # Valeur associée
for t, df in peaks_tab2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4]) 
        lat_idx = int(row[5]) 
        matrice_peaks_tab_p[t, lon_idx, lat_idx] = row[1] 

# max_lon = max(df[4].max() for df in val_peak2.values()) + 1
# max_lat = max(df[5].max() for df in val_peak2.values()) + 1
# nb_temps = len(val_peak2)

# Initialiser la matrice 3D avec NaN
matrice_val_peak_n = np.full((nb_temps, max_lon, max_lat), np.nan)
matrice_val_peak_p = np.full((nb_temps, max_lon, max_lat), np.nan)

for t, df in val_peak2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_val_peak_n[t, lon_idx, lat_idx] = row[0]
for t, df in val_peak2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_val_peak_p[t, lon_idx, lat_idx] = row[1]


# # Trouver les dimensions max en lon et lat
# max_lon = max(df[4].max() for df in peaks_freq2.values()) + 1
# max_lat = max(df[5].max() for df in peaks_freq2.values()) + 1
# nb_temps = len(peaks_freq2)

matrice_peaks_freq_n = np.full((nb_temps, max_lon, max_lat), np.nan)
matrice_peaks_freq_p = np.full((nb_temps, max_lon, max_lat), np.nan)

for t, df in peaks_freq2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_peaks_freq_n[t, lon_idx, lat_idx] = row[0]
for t, df in peaks_freq2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_peaks_freq_p[t, lon_idx, lat_idx] = row[1]

# max_lon = max(df[4].max() for df in peaks_freq2.values()) + 1
# max_lat = max(df[5].max() for df in peaks_freq2.values()) + 1
# nb_temps = len(peaks_freq2)

matrice_offset_n = np.full((nb_temps, max_lon, max_lat), np.nan)
matrice_offset_p = np.full((nb_temps, max_lon, max_lat), np.nan)

for t, df in peaks_freq2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_offset_n[t, lon_idx, lat_idx] = row[0] + f_B 
for t, df in peaks_freq2.items():
    for _, row in df.iterrows():
        lon_idx = int(row[4])
        lat_idx = int(row[5])
        matrice_offset_p[t, lon_idx, lat_idx] = row[1] - f_B

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.pcolormesh(matrice_ur[0,:,:], shading='auto')  # .T pour avoir le temps en x et les points en y
plt.pcolormesh(matrice_offset_n[0,:,:].T, shading='auto')
# plt.show()
    
#%% Création dictionnaire des matrices et teléchargement

ur_index = pd.to_datetime(time_ur)
pxy_index = pd.to_datetime(time_pxy)

index_existing = pxy_index.intersection(ur_index)

# Obtenir les positions sans erreur même si doublons :
pxy_indices = np.where(np.isin(pxy_index, index_existing))[0]
ur_indices = np.where(np.isin(ur_index, index_existing))[0]

matrice_peaks_tab_n = matrice_peaks_tab_n [pxy_indices,:,:]
matrice_peaks_tab_p = matrice_peaks_tab_p [pxy_indices,:,:]
matrice_val_peak_n = matrice_val_peak_n [pxy_indices,:,:]
matrice_val_peak_p = matrice_val_peak_p [pxy_indices,:,:]
matrice_peaks_freq_n = matrice_peaks_freq_n [pxy_indices,:,:]
matrice_peaks_freq_p = matrice_peaks_freq_p [pxy_indices,:,:]
matrice_offset_n = matrice_offset_n [pxy_indices,:,:]
matrice_offset_p = matrice_offset_p [pxy_indices,:,:]
matrice_ur = matrice_ur[ur_indices,:,:]

mean_per_layer = np.nanmean(matrice_ur, axis=(1, 2))

time_ur = pd.to_datetime(time_ur)
time_pxy = pd.to_datetime(time_pxy)
dict_data = {'matrice_peaks_tab_n' : matrice_peaks_tab_n,
             'matrice_peaks_tab_p' : matrice_peaks_tab_p,
             'matrice_val_peak_n' : matrice_val_peak_n,
             'matrice_val_peak_p' : matrice_val_peak_p,
             'matrice_peaks_freq_n' : matrice_peaks_freq_n,
             'matrice_peaks_freq_p' : matrice_peaks_freq_p,
             'matrice_offset_n' : matrice_offset_n,
             'matrice_offset_p' : matrice_offset_p,
             'matrice_ur' : matrice_ur.transpose(0, 2, 1),
             'points_lon_lat' : points_lon_lat,
             'coordonnes_lon_lat' : coordonnes_lon_lat,
             'time_ur' : ur_index,
             'time_pxy' : pxy_index}

import pickle 
# Sauvegarde du dictionnaire
with open('D:/Data_HFR/PAO/dict_data_PAO_2016.pkl', 'wb') as f:
    pickle.dump(dict_data, f)