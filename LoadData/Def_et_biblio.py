# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:57:19 2025

@author: geofd
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from tqdm import tqdm
import joblib
import pickle


def obtenir_variables_fichiers_netCDF(fichier):
    data = {}
    
    fichier_nc = nc.Dataset(fichier, "r")
    variables_fichier = [var for var in fichier_nc.variables.keys() ]

    for variable in variables_fichier:
        donnees = fichier_nc.variables[variable][:].data
        data[variable] = donnees  
        
    return data 
def create_temporal_features(df, columns, time_windows=['2H', '6H', '12H', '25H']):
    df_copy = df.copy()
    
    df_copy = df_copy.set_index('time')
    
    for column in columns:
        df_copy[f'{column}_last'] = df_copy[column].shift(1)
        
        for window in time_windows:
            df_copy[f'{column}_mean_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_copy[f'{column}_std_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .std()
            )
    
    # Reset index to keep 'time' as a column
    df_copy = df_copy.reset_index()
    return df_copy

def obtenir_variables_fichiers_netCDF(fichier):
    data = {}
    
    fichier_nc = nc.Dataset(fichier, "r")
    variables_fichier = [var for var in fichier_nc.variables.keys() ]

    for variable in variables_fichier:
        donnees = fichier_nc.variables[variable][:].data
        data[variable] = donnees  
        
    return data 
def create_temporal_features_vague(df, columns, time_windows=['2H', '6H', '12H', '25H']):
    df_copy = df.copy()
    
    df_copy = df_copy.set_index('time')
    
    for column in columns:
        df_copy[f'{column}_last'] = df_copy[column].shift(1)
        
        for window in time_windows:
            df_copy[f'{column}_mean_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_copy[f'{column}_std_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .std()
            )
    
    # Reset index to keep 'time' as a column
    df_copy = df_copy.reset_index()
    return df_copy

target_columns = ['VSDX', 'VSDY']

def prepare_time_series(df_X, df_yobs, target_columns, time_windows=['2H', '6H', '12H', '24H']):
    # Mise en forme des index temporels
    df_X = df_X_list[0].set_index('time')
    df_X.index = pd.to_datetime(df_X.index)
    
    df_yobs = df_yobs.set_index('time')
    df_yobs.index = pd.to_datetime(df_yobs.index)
    df_yobs = df_yobs.resample('5T').interpolate(method='linear')
    
    # Intersection des index
    index_existing = df_yobs.index.intersection(df_X.index)
    data = pd.concat([df_yobs.loc[index_existing], df_X.loc[index_existing]], axis=1)

    # Ajout des caractéristiques temporelles
    data = data.reset_index().rename(columns={'index': 'time'}).sort_values(by='time')
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data = pd.get_dummies(data, columns=['month', 'hour'])

    # Création des features temporelles
    new_data = create_temporal_features_vague(data, target_columns, time_windows)
    data = new_data.dropna()

    # Séparation des données
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    x_train, y_train = train_data.drop(labels=['time','VSDX', 'VSDY'], axis=1), train_data[['VSDX', 'VSDY']]
    x_test, y_test = test_data.drop(labels=['time','VSDX', 'VSDY'], axis=1), test_data[['VSDX', 'VSDY']]


    return x_train, y_train, x_test, y_test

def obtenir_variables_fichiers_netCDF(fichier):
    data = {}
    
    fichier_nc = nc.Dataset(fichier, "r")
    variables_fichier = [var for var in fichier_nc.variables.keys() ]

    for variable in variables_fichier:
        donnees = fichier_nc.variables[variable][:].data
        data[variable] = donnees  
        
    return data 

def create_temporal_features_courant(df, target_columns, time_windows):
    df_copy = df.copy()
    
    # df_copy = df_copy.set_index('time')
    
    for column in df_copy.columns:
        df_copy[f'{column}_last'] = df_copy[column].shift(1)
        
        for window in time_windows:
            df_copy[f'{column}_mean_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_copy[f'{column}_std_{window}'] = (
                df_copy[column]
                .shift()  # Shift first to exclude current value
                .rolling(window=window, min_periods=1)
                .std()
            )
    
    # Reset index to keep 'time' as a column
    df_copy = df_copy.reset_index()
    return df_copy

def prepare_time_series_courant(df_X, df_yobs, target_columns): # time_windows
    target_columns = ['uo','vo']
    time_windows=['2H', '6H', '12H', '24H']
    # Mise en forme des index temporels
    df_X = df_X.set_index('time')
    df_X.index = pd.to_datetime(df_X.index)
    # df_yobs = df_yobs_a
    df_yobs = df_yobs.set_index('time')
    df_yobs.index = pd.to_datetime(df_yobs.index)
    
    # Intersection des index
    index_existing = df_yobs.index.intersection(df_X.index)
    data = pd.concat([df_yobs.loc[index_existing], df_X.loc[index_existing]], axis=1)

    # Ajout des caractéristiques temporelles
    data = data.reset_index().rename(columns={'index': 'time'}).sort_values(by='time')
    data['month'] = data['time'].dt.month
    data['hour'] = data['time'].dt.hour
    data = pd.get_dummies(data, columns=['month', 'hour'])

    # Création des features temporelles
    new_data = create_temporal_features_courant(data, target_columns, time_windows)
    data = new_data.dropna()

    # Filtrage des valeurs spécifiques
    data = data[(data['uo'] != 0.134) & (data['vo'] != -0.181)]

    # Séparation des données
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    x_train, y_train = train_data.drop(labels=['time','uo','vo'], axis=1), train_data[['uo','vo']]
    x_test, y_test = test_data.drop(labels=['time','uo','vo'], axis=1), test_data[['uo','vo']]

    return x_train, y_train, x_test, y_test

def rmse(predictions, targets):  # Définition de la fonction RMSE
    return np.sqrt(((predictions - targets) ** 2).mean(axis=0))

def nse(predictions, targets):  # Définition de la fonction NSE
    return 1 - (np.sum((predictions - targets) ** 2, axis=0) / np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=0))

def kge(predictions, targets):  # Définition de la fonction KGE
    r = np.corrcoef(predictions.T, targets.T)[0,1]  # Assuming predictions and targets are 2D
    alpha = np.std(predictions, axis=0) / np.std(targets, axis=0)
    beta = np.mean(predictions, axis=0) / np.mean(targets, axis=0)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def index_of_agreement(predictions, targets):  # Définition de la fonction IA
    denominator = np.sum((np.abs(predictions - np.mean(targets, axis=0)) + np.abs(targets - np.mean(targets, axis=0)))**2, axis=0)
    numerator = np.sum((predictions - targets)**2, axis=0)
    return 1 - (numerator / denominator)


def create_temporal_features_vent(df, columns, time_windows=['2h', '6h', '12h', '24h']):
    df_copy = df.copy()
    
    # Vérifier que 'time' est bien en format datetime
    df_copy['time'] = pd.to_datetime(df_copy.index)
    
    # Assurer que 'time' est l'index
    df_copy = df_copy.set_index('time')
    
    for column in columns:
        # Dernière valeur connue (valeur précédente)
        df_copy[f'{column}_last'] = df_copy[column].shift(1)
        
        # Calcul des moyennes et écarts-types sur différentes fenêtres temporelles
        for window in time_windows:
            window_timedelta = pd.Timedelta(window)  # Convertir la fenêtre en Timedelta
            df_copy[f'{column}_mean_{window}'] = (
                df_copy[column]
                .shift()  # Décaler pour exclure la valeur courante
                .rolling(window=window_timedelta, min_periods=1)
                .mean()
            )
            df_copy[f'{column}_std_{window}'] = (
                df_copy[column]
                .shift()
                .rolling(window=window_timedelta, min_periods=1)
                .std()
            )
    
    # Réinitialiser l'index pour garder 'time' en colonne
    df_copy = df_copy.reset_index()
    
    return df_copy

def process_df_X_vent(df_X):
    target_columns = [col for col in df_X.columns if col != 'time']
    df_with_features = create_temporal_features_vent(df_X, target_columns)  # Fonction existante
    df_with_features.set_index('time', inplace=True)
    df_with_features.index = pd.to_datetime(df_with_features.index)
    
    return df_with_features


def inter(df_yobs, df_x):
    """Intersexe les index de df_yobs et df_x et retourne les DataFrames nettoyés."""
    index_existing = df_yobs.index.intersection(df_x.index)
    
    df_yobs_f = df_yobs.loc[index_existing]  # Correction ici
    df_X_f = df_x.loc[df_yobs_f.index]       # Assure l'alignement des index

    return df_X_f, df_yobs_f


# Fonction pour créer les features temporelles
features = ['peak_m', 'peak_p', 'vpeak_m',
       'vpeak_p', 'fpeak_m', 'fpeak_p',
       'offset_m', 'offset_p', 'Ur','Vitesse du vent', 'Direction du vent',
       'Vitesse du vent_last',
       'Vitesse du vent_mean_2h', 'Vitesse du vent_std_2h',
       'Vitesse du vent_mean_6h', 'Vitesse du vent_std_6h',
       'Vitesse du vent_mean_12h', 'Vitesse du vent_std_12h',
       'Vitesse du vent_mean_24h', 'Vitesse du vent_std_24h',
       'Direction du vent_last', 'Direction du vent_mean_2h',
       'Direction du vent_std_2h', 'Direction du vent_mean_6h',
       'Direction du vent_std_6h', 'Direction du vent_mean_12h',
       'Direction du vent_std_12h', 'Direction du vent_mean_24h',
       'Direction du vent_std_24h']

def create_temporal_features_vent2(df, columns, time_windows=['2h', '6h', '12h', '24h']):
    df_copy = df.copy()
    
    df_copy['time'] = pd.to_datetime(df_copy['time'])  # conversion si nécessaire
    df_copy = df_copy.set_index('time')
    
    for column in columns:
        df_copy[f'{column}_last'] = df_copy[column].shift(1)
        
        for window in time_windows:
            df_copy[f'{column}_mean_{window}'] = (
                df_copy[column]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
            df_copy[f'{column}_std_{window}'] = (
                df_copy[column]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .std()
            )
    
    df_copy = df_copy.reset_index()  # remettre time en colonne
    return df_copy[features]

def tret(extracted_data):
    df_concat_dict = {}
    time_index = pd.to_datetime(extracted_data['time_pxy'])
    
    # Boucle sur chaque matrice
    for key in matrices_to_flatten:
        # Création du DataFrame initial
        matrice3d = extracted_data[key]
        t, x, y = matrice3d.shape
        # matrice3d_points = matrice3d[:,np.array(lon_crad_iml4),np.array(lat_crad_iml4)]
        df = pd.DataFrame(matrice3d.reshape(t, x * y))
        # df.set_index(time_index, inplace=True)
    
        # Aplatir les valeurs (flatten)
        serie_concat = df.values.flatten()
    
        # Répéter l'index temporel pour correspondre au nombre de colonnes
        index_repeated = np.repeat(df.index, df.shape[1])
    
        # Créer le DataFrame final aplati
        df_concat_dict[key] = pd.DataFrame({
            'time': index_repeated,
            key: serie_concat
        })
    return df_concat_dict

matrices_to_flatten = [
    'matrice_peaks_tab_n',
    'matrice_peaks_tab_p',
    'matrice_val_peak_n',
    'matrice_val_peak_p',
    'matrice_peaks_freq_n',
    'matrice_peaks_freq_p',
    'matrice_offset_n',
    'matrice_offset_p',
    'matrice_ur'
]