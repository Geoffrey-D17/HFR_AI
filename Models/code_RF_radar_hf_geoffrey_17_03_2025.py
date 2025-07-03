# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:38:58 2025

@author: geofd
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection  import train_test_split
from tqdm import tqdm
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


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


def create_temporal_features(df, columns, time_windows=['2H', '6H', '12H', '24H']):
    df_copy = df.copy()
    
    # Vérifier que 'time' est bien en format datetime
    df_copy['time'] = pd.to_datetime(df_copy['time'])
    
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

def process_df_X(df_X):
    target_columns = [col for col in df_X.columns if col != 'time']
    df_with_features = create_temporal_features(df_X, target_columns)  # Fonction existante
    df_with_features.set_index('time', inplace=True)
    df_with_features.index = pd.to_datetime(df_with_features.index)
    
    return df_with_features

#%%

# Définition du chemin des fichiers
data_path = Path("Data/Points_PAO/")

# Générer automatiquement les noms de fichiers pour df_X
df_X_list = []
for i in range(88):
    file_path = data_path / f"point_vent_{i}_radar_PAO_2013.csv"
    if file_path.exists():
        df_X_list.append(pd.read_csv(file_path))
    else:
        print(f"Fichier manquant : {file_path.name}")

# Lecture des fichiers df_yobs
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
df_yobs_list[8].columns = ['time', 'Eastward Wind', 'Northward Wind']
# Prétraitement des df_yobs
for df_yobs in df_yobs_list:
    df_yobs.set_index('time', inplace=True)
    df_yobs.index = pd.to_datetime(df_yobs.index)

# Création des features et mise en index temporel
df_X = df_X_list[0]
df_X_processed_list = [process_df_X(df_X.drop(columns=['nb_point', 'lon', 'lat'])) for df_X in df_X_list]


# Concaténer toutes les données X
df_x_liste_pao = {
    'df_X_combined_a' : pd.concat(df_X_processed_list[:8], axis=0),
    'df_X_combined_b' : pd.concat(df_X_processed_list[8:16], axis=0),
    'df_X_combined_c' : pd.concat(df_X_processed_list[16:24], axis=0),
    'df_X_combined_d' : pd.concat(df_X_processed_list[24:32], axis=0), 
    'df_X_combined_e' : pd.concat(df_X_processed_list[32:40], axis=0),
    'df_X_combined_f' : pd.concat(df_X_processed_list[40:48], axis=0),
    'df_X_combined_g' : pd.concat(df_X_processed_list[48:56], axis=0),
    'df_X_combined_h' : pd.concat(df_X_processed_list[56:64], axis=0),
    'df_X_combined_i' : pd.concat(df_X_processed_list[64:72], axis=0), 
    'df_X_combined_j' : pd.concat(df_X_processed_list[72:80], axis=0)}
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
for df_y, df_x_key in zip(df_yobs_list, df_x_liste_pao):
    df_X_f_drop_n, df_yobs_f_drop_n = inter(df_y, df_x_liste_pao[df_x_key])

    # Stocker les résultats
    df_X_f_drop_n_list.append(df_X_f_drop_n)
    df_yobs_f_drop_n_list.append(df_yobs_f_drop_n)

    # Concaténation et ajout à la liste
    data = pd.concat([df_X_f_drop_n, df_yobs_f_drop_n], axis=1)
    data = data.dropna()
    data_liste.append(data)

# Concaténer les résultats finaux
data_liste_concat_pao = pd.concat(data_liste, axis=0)

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

df_X_processed_list = [process_df_X(df_X.drop(columns=['nb_point', 'lon', 'lat'])) for df_X in df_X_list]

# Concaténer toutes les données X
df_x_liste_pab = {
    'df_X_combined_a' : pd.concat(df_X_processed_list[:8], axis=0),
    'df_X_combined_b' : pd.concat(df_X_processed_list[16:24], axis=0),
    'df_X_combined_c' : pd.concat(df_X_processed_list[32:40], axis=0)}

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
for df_y, df_x_key in zip(df_yobs_list, df_x_liste_pab):
    df_X_f_drop_n, df_yobs_f_drop_n = inter(df_y, df_x_liste_pab[df_x_key])

    # Stocker les résultats
    df_X_f_drop_n_list.append(df_X_f_drop_n)
    df_yobs_f_drop_n_list.append(df_yobs_f_drop_n)

    # Concaténation et ajout à la liste
    data = pd.concat([df_X_f_drop_n, df_yobs_f_drop_n], axis=1)
    data = data.dropna()
    data_liste.append(data)

# Concaténer les résultats finaux
data_liste_concat_pab = pd.concat(data_liste, axis=0)

#%%
# Paramètres
initial_estimators = 1
increment = 1  # On augmente par 10 pour accélérer
total_estimators = 100

# Modèle avec warm_start=True pour un entraînement progressif
rf_model = RandomForestRegressor(n_estimators=initial_estimators, random_state=42, warm_start=True)

print("Entraînement du modèle...")

data_liste_concat = pd.concat([data_liste_concat_pao, data_liste_concat_pab], axis = 0)

X = data_liste_concat.drop(columns = ['Eastward Wind','Northward Wind']) # _drop_n
Y = data_liste_concat[['Eastward Wind','Northward Wind']]

columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle=True)
# Boucle d'entraînement
for i in tqdm(range(initial_estimators, total_estimators + 1, increment), desc="Entraînement du modèle", unit="incrément"):
    rf_model.n_estimators = i
    rf_model.fit(X_train, y_train)

# Sauvegarde finale
joblib.dump(rf_model, 'RF_vent.pkl')
print("Entraînement complet et modèle enregistré.")

# Prédiction sur les données alignées
y_pred = rf_model.predict(X_test)
y_true = y_test.values

y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_absolute_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")


#%%

print("Dimensions de y_true:", y_true.shape)
print("Dimensions de y_pred:", y_pred.shape)
y_pred_dt = pd.DataFrame(y_pred)

plt.figure(figsize=(20, 12))

# Taille des polices
label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

# Scatter plot of actual vs predicted (Eastward Wind)
plt.subplot(3, 2, 1)
plt.scatter(y_test['Eastward Wind'], y_pred_dt[0], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# coeffs = np.polyfit(y_test['Eastward Wind'], y_pred_dt[0], deg=1)  # deg=1 pour une droite
# poly_eq = np.poly1d(coeffs)
# plt.plot(y_test['Eastward Wind'], poly_eq(y_test['Eastward Wind']), color='red', label=f"Régression : y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")


# Time series plot of actual vs predicted
plt.subplot(3, 2, 3)
plt.plot(y_pred_dt.index, y_test['Eastward Wind'], label='Valeurs Réelles')
plt.plot(y_pred_dt.index, y_pred_dt[0], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 5)
errors = y_pred_dt[0].values - y_test['Eastward Wind'].values
sns.histplot(errors, kde=True, bins=20)
plt.xlabel('Erreur', fontsize=label_fontsize)
plt.ylabel('Fréquence', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Scatter plot of actual vs predicted (Northward Wind)
plt.subplot(3, 2, 2)
plt.scatter(y_test['Northward Wind'], y_pred_dt[1], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# coeffs = np.polyfit(y_test['Northward Wind'], y_pred_dt[1], deg=1)  # deg=1 pour une droite
# poly_eq = np.poly1d(coeffs)
# plt.plot(y_test['Northward Wind'], poly_eq(y_test['Northward Wind']), color='red', label=f"Régression : y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")

# Time series plot of actual vs predicted
plt.subplot(3, 2, 4)
plt.plot(y_pred_dt.index, y_test['Northward Wind'], label='Valeurs Réelles')
plt.plot(y_pred_dt.index, y_pred_dt[1], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 6)
errors = y_pred_dt[1].values - y_test['Northward Wind'].values
sns.histplot(errors, kde=True, bins=20)
plt.xlabel('Erreur', fontsize=label_fontsize)
plt.ylabel('Fréquence', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()

#%%
importances = rf_model.feature_importances_

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
sns.barplot(x=importances[indices], y = np.array(feature_names)[indices])
plt.xlabel("Importance", fontsize=label_fontsize)
plt.ylabel("Feature", fontsize=label_fontsize)
plt.title("Feature Importance Plot", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()


