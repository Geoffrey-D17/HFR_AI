# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:28:51 2025

@author: geofd
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


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

#%%

df_X_ = pd.read_csv('Data/point_1_radar_crad_PAB_2013.csv')
df_yobs = pd.read_csv('Data/point_cop_PAB_2013.csv')

df_X_.set_index('time', inplace=True)
df_yobs.set_index('time', inplace=True)


index_existing = df_yobs.index.intersection(df_X_.index)
df_yobs_f = df_yobs.loc[index_existing]
df_X_f = df_X_.loc[df_yobs_f.index]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # On prend la dernière sortie

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = df_X_f.shape[1]  
hidden_size = 128  # Moins de neurones que ANN car LSTM capture mieux les dépendances temporelles
output_size = 2  
num_layers = 5  
learning_rate = 1e-4  
num_epochs = 20  
batch_size = 1 

model_lstm = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=learning_rate)

# Transformation des données
train_dataset = TensorDataset(torch.tensor(df_X_f.values, dtype=torch.float32).unsqueeze(1), 
                              torch.tensor(df_yobs_f.values, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Entraînement du modèle
model_lstm.train()
for epoch in range(num_epochs):
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model_lstm(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

X_test = pd.read_csv('Data/point_1_radar_crad_PAB_2013.csv')
y_true =  pd.read_csv('Data/point_cop_PAB_2013.csv')
X_test.set_index('time', inplace = True)
y_true.set_index('time', inplace = True)

index_existing = y_true.index.intersection(X_test.index)
y_true_f = y_true.loc[index_existing]
X_test_f = X_test.loc[y_true_f.index]

# Passage en mode évaluation
model_lstm.eval()

# Création du Dataset et DataLoader pour le test
dataset = TensorDataset(torch.tensor(X_test_f.values, dtype=torch.float32).unsqueeze(1))  
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Liste pour stocker les prédictions
y_lstm_pred = []

# Désactivation du calcul des gradients
with torch.no_grad():
    for data in loader:
        data = data[0].to(device)  # Correction pour éviter les tuples
        outputs, _ = model_lstm.lstm(data)  # LSTM retourne une sortie + état caché
        final_output = model_lstm.fc(outputs[:, -1, :])  # Prendre la dernière sortie temporelle
        y_lstm_pred.append(final_output.cpu().numpy())  # Conversion en numpy

# Transformation en tableau numpy
y_lstm_pred = np.vstack(y_lstm_pred)

# Calcul et affichage des métriques
print("RMSE:", rmse(y_true_f.values, y_lstm_pred))  # Calcul du RMSE
print("NSE:", nse(y_true_f.values, y_lstm_pred))  # Calcul du NSE
print("KGE:", kge(y_true_f.values, y_lstm_pred))  # Calcul du KGE
print("IA:", index_of_agreement(y_true_f.values, y_lstm_pred))  # Calcul de l'index d'accord

print("Dimensions de y_true:", y_true_f.shape)
print("Dimensions de y_pred:", y_lstm_pred.shape)
y_pred_dt = pd.DataFrame(y_lstm_pred)

plt.figure(figsize=(20, 12))

# Scatter plot of actual vs predicted
plt.subplot(3, 2, 1)
plt.scatter(y_true_f['Eastward Wind'], y_pred_dt[0], alpha=0.5)
# plt.title(f'Comparaison des Prédictions et des Valeurs Réelles pour {output} RF')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

# Time series plot of actual vs predicted
plt.subplot(3, 2, 3)
plt.plot(pd.to_datetime(y_true_f.index),y_true_f['Eastward Wind'], label='Valeurs Réelles')
plt.plot(pd.to_datetime(y_true_f.index), y_pred_dt[0], label='Prédictions', linestyle='--')
# plt.title(f'Prédictions du Modèle vs Valeurs Réelles au Fil du Temps pour {output} RF')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()

# Distribution plot of prediction errors
plt.subplot(3, 2, 5)
errors = y_pred_dt[0].values - y_true_f['Eastward Wind'].values
sns.histplot(errors, kde=True, bins=20)
# plt.title(f'Distribution des Erreurs de Prédiction pour {output} RF')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')

plt.subplot(3, 2, 2)
plt.scatter(y_true_f['Northward Wind'], y_pred_dt[1], alpha=0.5)
# plt.title(f'Comparaison des Prédictions et des Valeurs Réelles pour {output} RF')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

# Time series plot of actual vs predicted
plt.subplot(3, 2, 4)
plt.plot(pd.to_datetime(y_true_f.index),y_true_f['Northward Wind'], label='Valeurs Réelles')
plt.plot(pd.to_datetime(y_true_f.index), y_pred_dt[1], label='Prédictions', linestyle='--')
# plt.title(f'Prédictions du Modèle vs Valeurs Réelles au Fil du Temps pour {output} RF')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()

# Distribution plot of prediction errors
plt.subplot(3, 2, 6)
errors = y_pred_dt[1].values - y_true_f['Northward Wind'].values
sns.histplot(errors, kde=True, bins=20)
# plt.title(f'Distribution des Erreurs de Prédiction pour {output} RF')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()