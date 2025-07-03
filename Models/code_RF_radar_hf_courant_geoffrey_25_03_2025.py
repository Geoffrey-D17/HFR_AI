# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:54:14 2025

@author: geofd
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from sklearn.linear_model import Ridge
from pathlib import Path
import joblib

def obtenir_variables_fichiers_netCDF(fichier):
    data = {}
    
    fichier_nc = nc.Dataset(fichier, "r")
    variables_fichier = [var for var in fichier_nc.variables.keys() ]

    for variable in variables_fichier:
        donnees = fichier_nc.variables[variable][:].data
        data[variable] = donnees  
        
    return data 
def create_temporal_features(df, target_columns, time_windows):
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

#%%

data_path = Path("Data/Points_PAO/")
df_X_list_pao = []
for i in range(16):
    file_path = data_path / f"point_courant_{i}_radar_PAO_2013.csv"  # Ajout de .csv
    if file_path.exists():  # Vérification si le fichier existe
        df_X_list_pao.append(pd.read_csv(file_path))

data_path = Path("Data/Points_PAB/")
df_X_list_pab = []
for i in range(16):
    file_path = data_path / f"point_courant_{i}_radar_PAB_2013.csv"  # Ajout de .csv
    if file_path.exists():  # Vérification si le fichier existe
        df_X_list_pab.append(pd.read_csv(file_path))
        
df_yobs_a = pd.read_csv("Data/Points_PAO/point_a_courant_cop_PAO_2013.csv")
df_yobs_b = pd.read_csv("Data/Points_PAO/point_b_courant_cop_PAO_2013.csv")
df_yobs_pab = pd.read_csv("Data/Points_PAB/point_courant_cop_PAB_2013.csv")
df_yobs_pab_ = df_yobs_pab[['time','uo_0','vo_0']]
df_yobs_pab_.columns = ['time','uo','vo']

target_columns = ['uo','vo']
time_windows=['2H', '6H', '12H', '24H']

def prepare_time_series(df_X, df_yobs, target_columns): # time_windows
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
    # data['month'] = data['time'].dt.month
    # data['hour'] = data['time'].dt.hour
    # data = pd.get_dummies(data, columns=['month', 'hour'])

    # Création des features temporelles
    # new_data = create_temporal_features(data, target_columns, time_windows)
    # data = new_data.dropna()

    # Filtrage des valeurs spécifiques
    data = data[(data['uo'] != 0.134) & (data['vo'] != -0.181)]

    # Séparation des données
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

    x_train, y_train = train_data.drop(labels=['time','uo','vo'], axis=1), train_data[['uo','vo']]
    x_test, y_test = test_data.drop(labels=['time','uo','vo'], axis=1), test_data[['uo','vo']]

    return x_train, y_train, x_test, y_test

x_train_list, y_train_list, x_test_list, y_test_list = zip(*[prepare_time_series(df_X, df_yobs_a, target_columns) for df_X in df_X_list_pao[:8]])
x_train_list_1, y_train_list_1, x_test_list_1, y_test_list_1 = zip(*[prepare_time_series(df_X, df_yobs_b, target_columns) for df_X in df_X_list_pao[8:]])
x_train_list_2, y_train_list_2, x_test_list_2, y_test_list_2 = zip(*[prepare_time_series(df_X, df_yobs_pab_, target_columns) for df_X in df_X_list_pab])

x_train_0, y_train_0 = pd.concat(x_train_list, axis = 0), pd.concat(y_train_list, axis = 0)
x_train_1, y_train_1 = pd.concat(x_train_list_1, axis = 0), pd.concat(y_train_list_1, axis = 0)
x_train_2, y_train_2 = pd.concat(x_train_list_2, axis = 0), pd.concat(y_train_list_2, axis = 0)
x_test_0, y_test_0 = pd.concat(x_test_list, axis = 0), pd.concat(y_test_list, axis = 0)
x_test_1, y_test_1 = pd.concat(x_test_list_1, axis = 0), pd.concat(y_test_list_1, axis = 0)
x_test_2, y_test_2 = pd.concat(x_test_list_2, axis = 0), pd.concat(y_test_list_2, axis = 0)

x_train, y_train = pd.concat([x_train_0, x_train_1, x_train_2], axis = 0), pd.concat([y_train_0, y_train_1, y_train_2], axis = 0)
x_test, y_test = pd.concat([x_test_0, x_test_1, x_test_2], axis = 0), pd.concat([y_test_0, y_test_1, y_test_2], axis = 0)
x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

x_train_clean = x_train.dropna()
y_train_clean = y_train.loc[x_train_clean.index]
x_test_clean = x_test.dropna()
y_test_clean = y_test.loc[x_test_clean.index]

x_train, y_train = x_train_clean.to_numpy().astype(float), y_train_clean.to_numpy().astype(float)
x_test, y_test = x_test_clean.to_numpy().astype(float), y_test_clean[['uo','vo']].to_numpy().astype(float)


#%%
model_Regressor = RandomForestRegressor(n_estimators=20, criterion='absolute_error')

model_Regressor.fit(x_train, y_train)
joblib.dump(model_Regressor, "random_forest_model.pkl")
y_pred_Regressor = model_Regressor.predict(x_test)
mae_Regressor = mean_absolute_error(y_test, y_pred_Regressor)
# rmse = root_mean_squared_error(y_test, y_pred)
print(f"MAE Regressor: {mae_Regressor:.3f}")

y_test = pd.DataFrame(y_test)
y_pred_Regressor_df = pd.DataFrame(y_pred_Regressor)
# y_pred_Embedding_df = pd.DataFrame(y_pred_Embedding)

y_pred_dt = pd.DataFrame(y_pred_Regressor_df)

plt.figure(figsize=(20, 12))

plt.subplot(3, 2, 1)
plt.scatter(y_test[0], y_pred_dt[0], alpha=0.5)
# plt.title(f'Comparaison des Prédictions et des Valeurs Réelles pour {output} RF')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

# Time series plot of actual vs predicted
plt.subplot(3, 2, 3)
plt.plot(y_test[0], label='Valeurs Réelles')
plt.plot(y_pred_dt[0], label='Prédictions', linestyle='--')
# plt.title(f'Prédictions du Modèle vs Valeurs Réelles au Fil du Temps pour {output} RF')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()

# Distribution plot of prediction errors
plt.subplot(3, 2, 5)
errors = y_pred_dt[0].values - y_test[0].values
sns.histplot(errors, kde=True, bins=20)
# plt.title(f'Distribution des Erreurs de Prédiction pour {output} RF')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')

plt.subplot(3, 2, 2)
plt.scatter(y_test[1], y_pred_dt[1], alpha=0.5)
# plt.title(f'Comparaison des Prédictions et des Valeurs Réelles pour {output} RF')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

# Time series plot of actual vs predicted
plt.subplot(3, 2, 4)
plt.plot(pd.to_datetime(y_test.index),y_test[1], label='Valeurs Réelles')
plt.plot(pd.to_datetime(y_test.index), y_pred_dt[1], label='Prédictions', linestyle='--')
# plt.title(f'Prédictions du Modèle vs Valeurs Réelles au Fil du Temps pour {output} RF')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()

# Distribution plot of prediction errors
plt.subplot(3, 2, 6)
errors = y_pred_dt[1].values - y_test[1].values
sns.histplot(errors, kde=True, bins=20)
# plt.title(f'Distribution des Erreurs de Prédiction pour {output} RF')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

criteria = ["squared_error", "absolute_error", "friedman_mse"]

# Stockage des résultats
results = []

# Boucle sur chaque critère
for criterion in criteria:
    print(f"Entraînement avec criterion = {criterion}")
    
    # Définition du modèle
    model = RandomForestRegressor(n_estimators=5, criterion=criterion, random_state=42)

    # Entraînement
    model.fit(x_train, y_train)

    # Prédiction
    y_pred = model.predict(x_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Stockage des résultats
    results.append({
        "Criterion": criterion,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    })

# Convertir en DataFrame pour affichage
df_results = pd.DataFrame(results)

# Afficher les résultats triés par MAE
print("\n Résultats des modèles triés par MAE :")
print(df_results.sort_values(by="MAE"))




