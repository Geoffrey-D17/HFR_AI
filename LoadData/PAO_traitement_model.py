# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:53:16 2025

@author: geofd
"""

from LoadData.Def_et_biblio import *
# from Loaddata.creation_data_liste import *

#%% traitement data PAO

with open("Data/PAB_PAO/dict_data_PAO_2013_v2.pkl", 'rb') as f:
    PAO2013 = pickle.load(f)
# with open("D:/Data_HFR/PAO/dict_data_PAO_2014_v2.pkl", 'rb') as f:
#     PAO2014 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAO_2015_v2.pkl", 'rb') as f:
    PAO2015 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAO_2016_v2.pkl", 'rb') as f:
    PAO2016 = pickle.load(f)
with open("Data/PAB_PAO/dict_data_PAO_2017_v2.pkl", 'rb') as f:
    PAO2017 = pickle.load(f)

data_all = []
for i in [PAO2013,  PAO2015, PAO2016, PAO2017]: #PAO2014,
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

df_X_pao = pd.concat(data_all, axis = 0)
df_X_pao.set_index('time', inplace = True)