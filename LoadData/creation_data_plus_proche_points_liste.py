# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:43:47 2025

@author: geofd
"""

from Loaddata.Def_et_biblio import *
from Loaddata.PAB_data import *
from Loaddata.PAO_data import *

target_lon_iml4 = -68.5833
target_lat_iml4 = 48.6667
target_lon_ws = -68.8928
target_lat_ws = 48.4156

lon_crad_ = u_r[0]['lon_crad']
lat_crad_ = u_r[0]['lat_crad']

# Si lon_array et lat_array sont des vecteurs 1D de même taille
dist = np.sqrt((lon_crad_ - target_lon_iml4)**2 + (lat_crad_ - target_lat_iml4)**2)
idx_closest_iml4 = np.argsort(dist)

peaks_freq2_ = []
val_peak2_ = []
peaks_tab2_ = []
u_r_ = []

for i in range(len(val_peak2)):
    length = len(u_r[i].values)
    valid_idx = [idx for idx in idx_closest_iml4[:10] if idx < length]
    
    u_r_.append(u_r[i].values[valid_idx])
    peaks_freq2_.append(peaks_freq2[i][peaks_freq2[i].index.isin(valid_idx)])
    val_peak2_.append(val_peak2[i][val_peak2[i].index.isin(valid_idx)])
    peaks_tab2_.append(peaks_tab2[i][peaks_tab2[i].index.isin(valid_idx)])

u_r_df_list = [pd.DataFrame(arr) for arr in u_r_]
peaks_freq2_list = [pd.DataFrame(arr) for arr in peaks_freq2_]
val_peak2_list = [pd.DataFrame(arr) for arr in val_peak2_]
peaks_tab2_list = [pd.DataFrame(arr) for arr in peaks_tab2_]

# Concaténer tous les DataFrames en un seul
u_r_concat = pd.concat(u_r_df_list, ignore_index=True)
peaks_freq2_concat = pd.concat(peaks_freq2_list, ignore_index=True)
val_peak2_concat = pd.concat(val_peak2_list, ignore_index=True)
peaks_tab2_concat = pd.concat(peaks_tab2_list, ignore_index=True)
u_r_concat = u_r_concat.dropna()

u_r_concat.set_index(pd.to_datetime(u_r_concat[2]), inplace = True)
peaks_freq2_concat.set_index(pd.to_datetime(peaks_freq2_concat[2]), inplace = True)
val_peak2_concat.set_index(pd.to_datetime(val_peak2_concat[2]), inplace = True)
peaks_tab2_concat.set_index(pd.to_datetime(peaks_tab2_concat[2]), inplace = True)

index_existing = peaks_tab2_concat.index.intersection(u_r_concat.index)

# Obtenir les positions sans erreur même si doublons :
peaks_freq2_concat_indice = np.where(np.isin(peaks_freq2_concat.index, index_existing))[0]
ur_indices = np.where(np.isin(u_r_concat.index, index_existing))[0]

u_r_concat_filt = u_r_concat.values[ur_indices]
peaks_freq2_concat = peaks_freq2_concat.values[peaks_freq2_concat_indice]
val_peak2_concat_filt = val_peak2_concat.values[peaks_freq2_concat_indice]
peaks_tab2_concat_filt = peaks_tab2_concat.values[peaks_freq2_concat_indice]

df_ur = pd.DataFrame(u_r_concat_filt)
df_peaks = pd.DataFrame(peaks_tab2_concat_filt)
df_vpeaks = pd.DataFrame(val_peak2_concat_filt)
df_fpeaks = pd.DataFrame(peaks_freq2_concat)

df_ur = df_ur.rename(columns={ 0: 'ur', 1: 'point', 2: 'time'})
df_peaks = df_peaks.rename(columns={0:'peak_m', 1:'peak_p', 2: 'time', 3: 'point'})
df_vpeaks = df_vpeaks.rename(columns={0:'vpeak_m', 1:'vpeak_p', 2: 'time', 3: 'point'})
df_fpeaks = df_fpeaks.rename(columns={0:'fpeak_m', 1:'fpeak_p', 2: 'time', 3: 'point'})

# Ensuite tu peux fusionner
merged_1 = pd.merge(df_ur[['ur','point','time']], df_peaks[['peak_m', 'peak_p','point','time']], on=['time', 'point'], how='inner')
merged_2 = pd.merge(merged_1, df_vpeaks[['vpeak_m','vpeak_p','point','time']], on=['time', 'point'], how='inner')
merged_all = pd.merge(merged_2, df_fpeaks[['fpeak_m','fpeak_p','point','time']], on=['time', 'point'], how='inner')

f_B = 0.4102
merged_all['offset_m'] = merged_all['fpeak_m'] + f_B
merged_all['offset_p'] = merged_all['fpeak_p'] - f_B

merged_all.to_csv('C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/PAO_IML4_2017.csv', index=False)