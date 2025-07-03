# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:51:23 2025

@author: geofd
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import mean_squared_error as mse

# Fonction pour créer les features temporelles
def create_temporal_features(df, columns, time_windows=['2H', '6H', '12H', '24H']):
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
    return df_copy

# Fonction de traitement principal des df_X
def process_df_X(df_X):
    target_columns = ['Eastward Wind', 'Northward Wind']

    # Ajout des colonnes mois et heure
    df_X['month'] = df_X.index.month
    df_X['hour'] = df_X.index.hour
    df_X['time'] = df_X.index
    df_with_features = create_temporal_features(df_X, target_columns)

    # Transformation en variables dummies
    df_with_features = pd.get_dummies(df_with_features, columns=['month', 'hour'])

    return df_with_features

# %% Chargement des données

# Chemin vers les fichiers
data_path = Path("Data/Points_PAO/")

# Chargement des fichiers radar df_X
df_X_list = []
for i in range(88):
    file_path = data_path / f"point_vent_{i}_radar_PAO_2013.csv"
    if file_path.exists():
        df_X_list.append(pd.read_csv(file_path))
    else:
        print(f"Fichier manquant : {file_path.name}")

# Chargement des fichiers d'observation df_yobs
df_yobs_files = [
    "point_a_cop_PAO_2013.csv",
    "point_b_cop_PAO_2013.csv",
    "point_c_cop_PAO_2013.csv",
    "point_d_cop_PAO_2013.csv",
    "point_e_cop_PAO_2013.csv",
    "point_f_cop_PAO_2013.csv",
    "point_g_cop_PAO_2013.csv",
    "point_h_cop_PAO_2013.csv",
    "point_i_cop_PAO_2013.csv",
    "point_j_cop_PAO_2013.csv",
    "point_k_cop_PAO_2013.csv"
]
df_yobs_list = [pd.read_csv(data_path.parent / file) for file in df_yobs_files]

# Harmonisation des noms de colonnes (si nécessaire)
df_yobs_list[8].columns = ['time', 'Eastward Wind', 'Northward Wind']

# Prétraitement des observations
for df_yobs in df_yobs_list:
    df_yobs['time'] = pd.to_datetime(df_yobs['time'])
    df_yobs.set_index('time', inplace=True)


# Concaténer toutes les données X
df_x_liste_pao = {
    'df_X_combined_a' : pd.concat(df_X_list[:8], axis=0),
    'df_X_combined_b' : pd.concat(df_X_list[8:16], axis=0),
    'df_X_combined_c' : pd.concat(df_X_list[16:24], axis=0),
    'df_X_combined_d' : pd.concat(df_X_list[24:32], axis=0), 
    'df_X_combined_e' : pd.concat(df_X_list[32:40], axis=0),
    'df_X_combined_f' : pd.concat(df_X_list[40:48], axis=0),
    'df_X_combined_g' : pd.concat(df_X_list[48:56], axis=0),
    'df_X_combined_h' : pd.concat(df_X_list[56:64], axis=0),
    'df_X_combined_i' : pd.concat(df_X_list[64:72], axis=0), 
    'df_X_combined_j' : pd.concat(df_X_list[72:80], axis=0)}
# df_X_combined_k = pd.concat(df_X_processed_list[80:], axis=0)

# Intersection entre df_yobs et df_X
df_X_f_drop_n_list = []
df_yobs_f_drop_n_list = []
data_liste = []

def inter(df_yobs, df_x):
    """Intersexe les index de df_yobs et df_x et retourne les DataFrames nettoyés."""
    index_existing = df_yobs.index.intersection(df_x.index)
    
    df_yobs_f = df_yobs.loc[index_existing]  # Correction ici
    df_X_f = df_x.loc[df_yobs_f.index]       # Assure l'alignement des index

    return df_X_f, df_yobs_f

# Intersection et fusion des données
for df_y, (df_x_name, df_x) in zip(df_yobs_list, df_x_liste_pao.items()):
    df_x.index = pd.to_datetime(df_x.index)
    # df_x.set_index('time', inplace = True)
    df_X_f_drop_n, df_yobs_f_drop_n = inter(df_y, df_x)

    # Stocker les résultats
    df_X_f_drop_n_list.append(df_X_f_drop_n)
    df_yobs_f_drop_n_list.append(df_yobs_f_drop_n)
    
    # Concaténation et ajout à la liste
    data = pd.concat([df_X_f_drop_n, df_yobs_f_drop_n], axis=1)
    data = data.dropna()
    data_liste.append(data)

# Création des features temporelles pour df_X_list
new_data_liste = []
for df_X in data_liste:
    df_X = df_X.drop(columns=['nb_point', 'lon', 'lat'])
    df_X['time'] = df_X.index
    df_X['month'] = df_X['time'].dt.month
    df_X['hour'] = df_X['time'].dt.hour
    df_X = pd.get_dummies(df_X, columns=['month', 'hour'])
    target_columns = ['Eastward Wind', 'Northward Wind']
    new_data = create_temporal_features(
    df_X, 
    target_columns,
    time_windows=['2H', '6H', '12H', '24H']
    )
    new_data_liste.append(new_data)

# Concaténer les résultats finaux
data_liste_concat_pao = pd.concat(new_data_liste, axis=0)

#%%

# Définition du chemin des fichiers
data_path = Path("Data/Points_PAB/")

# Générer automatiquement les noms de fichiers pour df_X
df_X_list = []
for i in range(48):
    file_path = data_path / f"point_{i}_radar_crad_PAB_2013.csv"
    if file_path.exists():
        df_X_list.append(pd.read_csv(file_path))
    else:
        print(f"Fichier manquant : {file_path.name}")

# Lecture des fichiers df_yobs
df_yobs_files = [
    "point_a_cop_PAB_2013.csv",
    "point_b_cop_PAB_2013.csv",
    "point_c_cop_PAB_2013.csv"
]
df_yobs_list = [pd.read_csv(data_path.parent / file) for file in df_yobs_files]
# df_yobs_list[8].columns = ['time', 'Eastward Wind', 'Northward Wind']
# Prétraitement des df_yobs
for df_yobs in df_yobs_list:
    df_yobs.set_index('time', inplace=True)
    df_yobs.index = pd.to_datetime(df_yobs.index)

# df_X_processed_list = [process_df_X(df_X.drop(columns=['nb_point', 'lon', 'lat'])) for df_X in df_X_list]

# Concaténer toutes les données X
df_x_liste_pab = {
    'df_X_combined_a' : pd.concat(df_X_list[:8], axis=0),
    'df_X_combined_b' : pd.concat(df_X_list[16:24], axis=0),
    'df_X_combined_c' : pd.concat(df_X_list[32:40], axis=0)}

# Intersection entre df_yobs et df_X
df_X_f_drop_n_list = []
df_yobs_f_drop_n_list = []
data_liste = []

# Intersection et fusion des données
for df_y, (df_x_name, df_x) in zip(df_yobs_list, df_x_liste_pab.items()):
    df_x.set_index('time', inplace = True)
    df_x.index = pd.to_datetime(df_x.index)
    # df_x.set_index('time', inplace = True)
    df_X_f_drop_n, df_yobs_f_drop_n = inter(df_y, df_x)

    # Stocker les résultats
    df_X_f_drop_n_list.append(df_X_f_drop_n)
    df_yobs_f_drop_n_list.append(df_yobs_f_drop_n)
    
    # Concaténation et ajout à la liste
    data = pd.concat([df_X_f_drop_n, df_yobs_f_drop_n], axis=1)
    data = data.dropna()
    data_liste.append(data)
    
new_data_liste = []
for df_X in data_liste:
    df_X = df_X.drop(columns=['nb_point', 'lon', 'lat'])
    df_X['time'] = df_X.index
    df_X['month'] = df_X['time'].dt.month
    df_X['hour'] = df_X['time'].dt.hour
    df_X = pd.get_dummies(df_X, columns=['month', 'hour'])
    target_columns = ['Eastward Wind', 'Northward Wind']
    new_data = create_temporal_features(
    df_X, 
    target_columns,
    time_windows=['2H', '6H', '12H', '24H']
    )
    new_data_liste.append(new_data)
    
# Concaténer les résultats finaux
data_liste_concat_pab = pd.concat(new_data_liste, axis=0)

#%%

features = ['Eastward Wind', 'Northward Wind', 'amp_neg', 'amp_pos',
       'offset_fb_neg', 'offset_fb_pos', 'freq_neg', 'freq_pos', 'Ur',
         'Eastward Wind_last',
       'Eastward Wind_mean_2H', 'Eastward Wind_std_2H',
       'Eastward Wind_mean_6H', 'Eastward Wind_std_6H',
       'Eastward Wind_mean_12H', 'Eastward Wind_std_12H',
       'Eastward Wind_mean_24H', 'Eastward Wind_std_24H',
       'Northward Wind_last', 'Northward Wind_mean_2H',
       'Northward Wind_std_2H', 'Northward Wind_mean_6H',
       'Northward Wind_std_6H', 'Northward Wind_mean_12H',
       'Northward Wind_std_12H', 'Northward Wind_mean_24H',
       'Northward Wind_std_24H']

data_pao = data_liste_concat_pao[features]
data_pab = data_liste_concat_pab[features]
data = pd.concat([data_liste_concat_pao, data_liste_concat_pab], axis=0)
data = data.dropna()

#%%

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle= True)
x_train, y_train = train_data.drop(labels=['Eastward Wind', 'Northward Wind'], axis=1), train_data[['Eastward Wind', 'Northward Wind']]
x_test, y_test = test_data.drop(labels=['Eastward Wind', 'Northward Wind'], axis=1), test_data[['Eastward Wind', 'Northward Wind']]

if 'time' in x_train.columns:
    x_train = x_train.drop(columns='time')
if 'time' in x_test.columns:
    x_test = x_test.drop(columns='time')

x_train, y_train = x_train.to_numpy().astype(float), y_train.to_numpy().astype(float)
x_test, y_test = x_test.to_numpy().astype(float), y_test.to_numpy().astype(float)

#%%

model = RandomForestRegressor(n_estimators=10, criterion='absolute_error')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_absolute_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")


#%%

print("Dimensions de y_true:", y_test.shape)
print("Dimensions de y_pred:", y_pred.shape)
y_pred_dt = pd.DataFrame(y_pred)

plt.figure(figsize=(20, 12))

# Taille des polices
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

# Scatter plot of actual vs predicted (Eastward Wind)
plt.subplot(3, 2, 1)
plt.scatter(y_test[:,0], y_pred[:,0], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Time series plot of actual vs predicted
plt.subplot(3, 2, 3)
plt.plot(y_test[:,0], label='Valeurs Réelles')
plt.plot(y_pred[:,0], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 5)
errors = y_pred[:,0] - y_test[:,0]
sns.histplot(errors, kde=True, bins=40)
plt.xlabel('Erreur', fontsize=label_fontsize)
plt.ylabel('Fréquence', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Scatter plot of actual vs predicted (Northward Wind)
plt.subplot(3, 2, 2)
plt.scatter(y_test[:,1], y_pred[:,1], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)


# Time series plot of actual vs predicted
plt.subplot(3, 2, 4)
plt.plot(y_test[:,1], label='Valeurs Réelles')
plt.plot(y_pred[:,1], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 6)
errors = y_pred[:,1]- y_test[:,1]
sns.histplot(errors, kde=True, bins=40)
plt.xlabel('Erreur', fontsize=label_fontsize)
plt.ylabel('Fréquence', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()

#%%
importances = model.feature_importances_

feature_names = ['amp_neg', 'amp_pos', 'offset_fb_neg', 'offset_fb_pos', 'freq_neg',
       'freq_pos', 'Ur', 'amp_neg_last', 'amp_neg_mean_2H', 'amp_neg_std_2H',
       'amp_neg_mean_6H', 'amp_neg_std_6H', 'amp_neg_mean_12H',
       'amp_neg_std_12H', 'amp_neg_mean_24H', 'amp_neg_std_24H',
       'amp_pos_last', 'amp_pos_mean_2H', 'amp_pos_std_2H', 'amp_pos_mean_6H',
       'amp_pos_std_6H', 'amp_pos_mean_12H', 'amp_pos_std_12H',
       'amp_pos_mean_24H', 'amp_pos_std_24H', 'offset_fb_neg_last',
       'offset_fb_neg_mean_2H', 'offset_fb_neg_std_2H',
       'offset_fb_neg_mean_6H', 'offset_fb_neg_std_6H',
       'offset_fb_neg_mean_12H', 'offset_fb_neg_std_12H',
       'offset_fb_neg_mean_24H', 'offset_fb_neg_std_24H', 'offset_fb_pos_last',
       'offset_fb_pos_mean_2H', 'offset_fb_pos_std_2H',
       'offset_fb_pos_mean_6H', 'offset_fb_pos_std_6H',
       'offset_fb_pos_mean_12H', 'offset_fb_pos_std_12H',
       'offset_fb_pos_mean_24H', 'offset_fb_pos_std_24H', 'freq_neg_last',
       'freq_neg_mean_2H', 'freq_neg_std_2H', 'freq_neg_mean_6H',
       'freq_neg_std_6H', 'freq_neg_mean_12H', 'freq_neg_std_12H',
       'freq_neg_mean_24H', 'freq_neg_std_24H', 'freq_pos_last',
       'freq_pos_mean_2H', 'freq_pos_std_2H', 'freq_pos_mean_6H',
       'freq_pos_std_6H', 'freq_pos_mean_12H', 'freq_pos_std_12H',
       'freq_pos_mean_24H', 'freq_pos_std_24H', 'Ur_last', 'Ur_mean_2H',
       'Ur_std_2H', 'Ur_mean_6H', 'Ur_std_6H', 'Ur_mean_12H', 'Ur_std_12H',
       'Ur_mean_24H', 'Ur_std_24H']

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(18, 15))
sns.barplot(x=importances[indices][:10], y = np.array(feature_names)[indices][:10])
plt.xlabel("Importance", fontsize=label_fontsize)
plt.ylabel("Feature", fontsize=label_fontsize)
plt.title("Feature Importance Plot", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()
