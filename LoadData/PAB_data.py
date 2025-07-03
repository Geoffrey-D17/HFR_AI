# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:47:49 2025

@author: geofd
"""

from Loaddata.Def_et_biblio import *

coo_pxy = pd.read_csv('C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Radar HF/WERA/LSLE_grid.asc', sep = '\s+')

points_lon_lat = pd.read_csv('C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Points_PAB/lon_lat_points_PAB.csv')
coordonnes_lon_lat = pd.read_csv('C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Points_PAB/lon_lat_coordonnees_PAB.csv')
freq = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Radar HF/WERA_2/freq.txt", header = None)
PXY_files = [f for f in os.listdir('D:/Data_HFR/PAB/SPEC_mat/')if f.endswith(".mat") and f.startswith("2020")]
Ur_files = [f for f in os.listdir("D:/Data_HFR/PAB/CRAD_mat2/")if f.endswith(".mat") and f.startswith("2020")]
freq = freq[0]

