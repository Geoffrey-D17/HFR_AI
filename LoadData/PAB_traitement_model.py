# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:52:05 2025

@author: geofd
"""

from LoadData.Def_et_biblio import *
# from Loaddata.creation_data_liste import *

#%% traitement data PAB

with open("Data/PAB_PAO/dict_data_PAB_2013_v2.pkl", 'rb') as f:
    pab2013 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAB_2014_v2.pkl", 'rb') as f:
    pab2014 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAB_2015_v2.pkl", 'rb') as f:
    pab2015 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAB_2016_v2.pkl", 'rb') as f:
    pab2016 = pickle.load(f)
# with open("Data/PAB_PAO/dict_data_PAB_2018_v2.pkl", 'rb') as f:
#     pab2018 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAB_2019_v2.pkl", 'rb') as f:
    pab2019 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAB_2020_v2.pkl", 'rb') as f:
    pab2020 = pickle.load(f)

data_all = []
for i in [pab2013, pab2014, pab2015, pab2016, pab2019, pab2020]: # pab2018,
    spec = pd.DataFrame(i['spec_data'])
    spec = spec.dropna()
    spec = spec.loc[:, ~spec.columns.duplicated()]
    # Reconvertis la colonne 'time' en datetime si nécessaire
    spec['time'] = pd.to_datetime(spec['time'])
    
    # Réassigne la colonne 'time' comme index
    # spec = spec.set_index('time')
    
    crad = pd.DataFrame(i['crad_data'])
    crad = crad.dropna()
    crad['point'] = crad.nb_point
    # crad.set_index('time', inplace = True)
    
    # index_existing = crad.index.intersection(spec.index)
    
    merged = pd.merge(spec, crad, on=['time', 'point'], how='inner') 
    data = merged[['fpeak_m', 'fpeak_p', 'time', 'vpeak_m', 'vpeak_p', 'peak_m', 'peak_p',
           'point', 'lon_p', 'lat_p', 'offset_m', 'offset_p', 'Ur',
           'lon_crad', 'lat_crad', 'lon_spec', 'lat_spec']]
    data_all.append(data)

df_X_pab = pd.concat(data_all, axis = 0)
df_X_pab.set_index('time', inplace = True)