# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:23:13 2025

@author: geofd
"""

from LoadData.Def_et_biblio import *
from LoadData.PAB_traitement_model import *
from LoadData.PAO_traitement_model import *

#%%

df_X = pd.concat([df_X_pab, df_X_pao], axis = 0)

#%% WS_BIC

df = pd.read_csv("Data/data_ref/WS_BIC_2014_2015.csv", encoding='latin1')
df2 = pd.read_csv("Data/data_ref/ws_bic_2016_2017.csv", encoding='latin1')
df = pd.concat([df,df2], axis = 0)
# Conversion de la date en datetime
df['date'] = pd.to_datetime(df['date'].str.replace('Z\[UTC\]', '', regex=True), utc = False)
# df['date'] = pd.to_datetime(df['date']) 
df_yobs = df[['date', 'Vitesse du vent', 'Direction du vent']]
df_yobs.set_index('date', inplace=True)
df_yobs_15min = df_yobs.resample('15T').interpolate()

# df_X.index = df_X.index.tz_localize('UTC')
df_X_f_drop_n, df_yobs_f_drop_n = inter(df_yobs_15min, df_X)
df_X_f_drop_n = df_X_f_drop_n[['fpeak_m', 'fpeak_p', 'vpeak_m', 'vpeak_p', 'peak_m', 'peak_p','offset_m', 'offset_p', 'Ur']]

df_merged = df_X_f_drop_n.merge(df_yobs_f_drop_n, left_index=True, right_index=True, how='left')
df_merged_WS_BIC = df_merged.dropna()

df_X1 = df_merged_WS_BIC
df_X1['time'] = df_X1.index
df_X1['month'] = df_X1['time'].dt.month
df_X1['hour'] = df_X1['time'].dt.hour
df_X1 = pd.get_dummies(df_X1, columns=['month', 'hour'])
target_columns = ['Vitesse du vent', 'Direction du vent']
df_merged_WS_BIC_2 = create_temporal_features2(df_X1, target_columns,time_windows=['2h', '6h', '12h', '24h'])


#%% IML4_RIKI

df = pd.read_csv("Data/Data_bouee/IML4_RIKI_2014_2017.csv", encoding='latin1')
# Conversion de la date en datetime
# df['date'] = pd.to_datetime(df['date'].str.replace('Z\[UTC\]', '', regex=True))
df['date'] = pd.to_datetime(df['date'].str.replace(r'Z\[UTC\]', '', regex=True),utc=False, errors='coerce')
# df['date'] = pd.to_datetime(df['date']) 
df_yobs = df[['date', 'Vitesse du vent', 'Direction du vent']]
df_yobs.set_index('date', inplace=True)
df_yobs = df_yobs[~df_yobs.index.duplicated(keep='first')]
df_yobs_15min = df_yobs.resample('1min').interpolate()
# df_X.index = df_X.index.tz_localize('UTC')
df_X_f_drop_n, df_yobs_f_drop_n = inter(df_yobs_15min, df_X)

df_merged = df_X_f_drop_n.merge(df_yobs_f_drop_n, left_index=True, right_index=True, how='left')
df_merged_IML4_RIKI = df_merged.dropna()

df_X2 = df_merged_IML4_RIKI
df_X2['time'] = df_X2.index
df_X2['month'] = df_X2['time'].dt.month
df_X2['hour'] = df_X2['time'].dt.hour
df_X2 = pd.get_dummies(df_X2, columns=['month', 'hour'])
target_columns = ['Vitesse du vent', 'Direction du vent']
df_merged_IML4_RIKI_2 = create_temporal_features2(df_X2, target_columns,time_windows=['2h', '6h', '12h', '24h'])


#%% model WS_BIC_1
# Paramètres
initial_estimators = 1
increment = 1  # On augmente par 10 pour accélérer
total_estimators = 10

# Modèle avec warm_start=True pour un entraînement progressif
rf_model = RandomForestRegressor(n_estimators=initial_estimators, random_state=42, warm_start=True)

Xbis = df_merged_WS_BIC.drop(columns = ['Vitesse du vent', 'Direction du vent', 'time', 'month', 'hour']) # _drop_n
X = process_df_X(Xbis)
Y = df_merged_WS_BIC[['Vitesse du vent', 'Direction du vent']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle=True)
WS_BIC_index = y_test.index
# Boucle d'entraînement
for i in tqdm(range(initial_estimators, total_estimators + 1, increment), desc="Entraînement du modèle", unit="incrément"):
    rf_model.n_estimators = i
    rf_model.fit(X_train, y_train)

# Sauvegarde finale
# joblib.dump(rf_model, 'ERA5_1_v1.pkl')

# Prédiction sur les données alignées
y_pred = rf_model.predict(X_test)
y_true = y_test.values
test_WS_BIC = y_test

y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_squared_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")

pred_WS_BIC_1 = pd.DataFrame(y_pred)
pred_WS_BIC_1.set_index(y_test.index, inplace = True)
pred_WS_BIC_1.columns = ['Vitesse', 'Direction']
pred_WS_BIC_1.sort_index(ascending=True, inplace = True)

#%% model IML4_RIKI_1

rf_model = RandomForestRegressor(n_estimators=initial_estimators, random_state=42, warm_start=True)

Xbis = df_merged_IML4_RIKI.drop(columns = ['Vitesse du vent', 'Direction du vent']) # _drop_n
X = process_df_X(Xbis)
Y = df_merged_IML4_RIKI[['Vitesse du vent', 'Direction du vent']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle=True)
IML4_RIKI_index = y_test.index
# Boucle d'entraînement
for i in tqdm(range(initial_estimators, total_estimators + 1, increment), desc="Entraînement du modèle", unit="incrément"):
    rf_model.n_estimators = i
    rf_model.fit(X_train, y_train)

# Sauvegarde finale
# joblib.dump(rf_model, 'ERA5_1_v1.pkl')

# Prédiction sur les données alignées
y_pred = rf_model.predict(X_test)
y_true = y_test.values

y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_squared_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")
# y_pred_dt = pd.DataFrame(y_test)
# y_pred_dt.set_index(y_test.index, inplace = True)
# y_pred_dt.to_csv('C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/test_2017_ws_bic_v1.csv', index=True)

pred_IML4_RIKI_1 = pd.DataFrame(y_pred)
pred_IML4_RIKI_1.set_index(y_test.index, inplace = True)
pred_IML4_RIKI_1.columns = ['Vitesse', 'Direction']
pred_IML4_RIKI_1.sort_index(ascending=True, inplace = True)

y_test.sort_index(ascending=True, inplace = True)
test_IML4_RIKI = y_test

plt.figure(figsize=(18, 15))
plt.plot(pred_IML4_RIKI_1['Vitesse'], color = 'red')
plt.plot(pred_WS_BIC_1['Vitesse'], color = 'blue')
plt.plot(y_test['Vitesse du vent'], color = 'black')

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 15))
plt.plot(pred_IML4_RIKI_1['Direction'], color = 'red')
plt.plot(pred_WS_BIC_1['Direction'], color = 'blue')
plt.plot(y_test['Direction du vent'], color = 'black')

plt.tight_layout()
plt.show()

#%% model WS_BIC_2
total_estimators = 10

df_merged_WS_BIC_2.set_index(df_merged_WS_BIC.index, inplace = True)
df_merged_WS_BIC_2 = df_merged_WS_BIC_2.dropna()

train_data, test_data = train_test_split(df_merged_WS_BIC_2, test_size=0.2, random_state=42, shuffle= True)
x_train, y_train = train_data.drop(labels=['Vitesse du vent', 'Direction du vent'], axis=1), train_data[['Vitesse du vent', 'Direction du vent']]
x_test, y_test = test_data.drop(labels=['Vitesse du vent', 'Direction du vent'], axis=1), test_data[['Vitesse du vent', 'Direction du vent']]

if 'time' in x_train.columns:
    x_train = x_train.drop(columns='time')
if 'time' in x_test.columns:
    x_test = x_test.drop(columns='time')

model = RandomForestRegressor(n_estimators=10, criterion='absolute_error')

for i in tqdm(range(initial_estimators, total_estimators + 1, increment), desc="Entraînement du modèle", unit="incrément"):
    model.n_estimators = i
    model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_squared_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")

pred_WS_BIC_2 = pd.DataFrame(y_pred)
pred_WS_BIC_2.set_index(y_test.index, inplace = True)
pred_WS_BIC_2.columns = ['Vitesse', 'Direction']
pred_WS_BIC_2.sort_index(ascending=True, inplace = True)

#%% model IML4_RIKI_2

df_merged_IML4_RIKI_2.set_index(df_merged_IML4_RIKI.index, inplace = True)
df_merged_IML4_RIKI_2 = df_merged_IML4_RIKI_2.dropna()

train_data, test_data = train_test_split(df_merged_IML4_RIKI_2, test_size=0.2, random_state=42, shuffle= True)
x_train, y_train = train_data.drop(labels=['Vitesse du vent', 'Direction du vent'], axis=1), train_data[['Vitesse du vent', 'Direction du vent']]
x_test, y_test = test_data.drop(labels=['Vitesse du vent', 'Direction du vent'], axis=1), test_data[['Vitesse du vent', 'Direction du vent']]

if 'time' in x_train.columns:
    x_train = x_train.drop(columns='time')
if 'time' in x_test.columns:
    x_test = x_test.drop(columns='time')

# x_train, y_train = x_train.to_numpy().astype(float), y_train.to_numpy().astype(float)
# x_test, y_test = x_test.to_numpy().astype(float), y_test.to_numpy().astype(float)
total_estimators = 10
model_IML4_RIKI_2 = RandomForestRegressor(n_estimators=10, criterion='absolute_error')
for i in tqdm(range(initial_estimators, total_estimators + 1, increment), desc="Entraînement du modèle", unit="incrément"):
    model_IML4_RIKI_2.n_estimators = i
    model_IML4_RIKI_2.fit(x_train, y_train)
    
# # Sauvegarde finale
# joblib.dump(model, 'IML4_RIKI_2_v2.pkl')

y_pred = model_IML4_RIKI_2.predict(x_test)
y_test_std = (y_test-y_test.mean())/y_test.std()
y_pred_std = (y_pred-y_pred.mean())/y_pred.std()
mae = mean_absolute_error(y_test_std, y_pred_std)
rmse = root_mean_squared_error(y_test_std, y_pred_std)
mse_ = mean_squared_error(y_test_std, y_pred_std)
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse_:.3f}")
print(f"RMSE: {rmse:.3f}")

pred_IML4_RIKI_2 = pd.DataFrame(y_pred)
pred_IML4_RIKI_2.set_index(y_test.index, inplace = True)
pred_IML4_RIKI_2.columns = ['Vitesse', 'Direction']
pred_IML4_RIKI_2.sort_index(ascending=True, inplace = True)

#%% plot comaraison model


test_IML4_RIKI = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/test_2017_v1.csv")
test_IML4_RIKI.set_index('Unnamed: 0', inplace = True)
test_IML4_RIKI.index = pd.to_datetime(test_IML4_RIKI.index)
test_WS_BIC = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/test_2017_ws_bic_v1.csv")
test_WS_BIC.set_index('Unnamed: 0', inplace = True)
test_WS_BIC.index = pd.to_datetime(test_WS_BIC.index)
pred_WS_BIC_1 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_WS_BIC_1_v1.csv")
pred_WS_BIC_1.set_index('Unnamed: 0', inplace = True)
pred_WS_BIC_1.index = pd.to_datetime(pred_WS_BIC_1.index)
pred_WS_BIC_1.columns = ["vent", 'dir']
pred_WS_BIC_2 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_WS_BIC_2_v1.csv")
pred_WS_BIC_2.set_index('date', inplace = True)
pred_WS_BIC_2.index = pd.to_datetime(pred_WS_BIC_2.index)
pred_WS_BIC_2.columns = ["vent", 'dir']
pred_IML4_RIKI_1 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_IML4_RIKI_1_v1.csv")
pred_IML4_RIKI_1.set_index('Unnamed: 0', inplace = True)
pred_IML4_RIKI_1.index = pd.to_datetime(pred_IML4_RIKI_1.index)
pred_IML4_RIKI_1.columns = ["vent", 'dir']
pred_IML4_RIKI_2 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_IML4_RIKI_2_v1.csv")
pred_IML4_RIKI_2.set_index('date', inplace = True)
pred_IML4_RIKI_2.index = pd.to_datetime(pred_IML4_RIKI_2.index)
pred_IML4_RIKI_2.columns = ["vent", 'dir']

test_IML4_RIKI.sort_index(ascending=True, inplace = True)
test_WS_BIC.sort_index(ascending=True, inplace = True)
pred_WS_BIC_1.sort_index(ascending=True, inplace = True)
pred_WS_BIC_2.sort_index(ascending=True, inplace = True)
pred_IML4_RIKI_1.sort_index(ascending=True, inplace = True)
pred_IML4_RIKI_2.sort_index(ascending=True, inplace = True)

label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

fig, axs = plt.subplots(4, 1, figsize=(20, 14), sharex=True)

# --- Vitesse du vent ---
axs[0].plot(pred_IML4_RIKI_1['Vitesse'], '--r', linewidth=3, label='IML4_RIKI_1')
axs[0].plot(pred_IML4_RIKI_2['Vitesse'], '--b', linewidth=3, label='IML4_RIKI_2')
axs[0].plot(test_IML4_RIKI['Vitesse du vent'], 'k', linewidth=2, label='Test IML4_RIKI')
axs[0].set_ylabel("Wind Speed (m/s)", fontsize=label_fontsize)
axs[0].tick_params(labelsize=tick_fontsize)
axs[0].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[0].grid(True)

# --- Direction du vent ---
axs[1].plot(pred_IML4_RIKI_1['Direction'], '--r', linewidth=3, label='IML4_RIKI_1')
axs[1].plot(pred_IML4_RIKI_2['Direction'], '--b', linewidth=3, label='IML4_RIKI_2')
axs[1].plot(test_IML4_RIKI['Direction du vent'],  'k', linewidth=2, label='Test IML4_RIKI')
axs[1].set_xlabel("Time", fontsize=label_fontsize)
axs[1].set_ylabel("Direction (°)", fontsize=label_fontsize)
axs[1].tick_params(labelsize=tick_fontsize)
axs[1].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[1].grid(True)

axs[2].plot(pred_WS_BIC_1['Vitesse'], '--r', linewidth=3, label='WS_BIC_1')
axs[2].plot(pred_WS_BIC_2['Vitesse'], '--b', linewidth=3, label='WS_BIC_2')
axs[2].plot(test_WS_BIC['Vitesse du vent'], 'k', linewidth=2, label='Test WS_BIC')
axs[2].set_ylabel("Wind Speed (m/s)", fontsize=label_fontsize)
axs[2].tick_params(labelsize=tick_fontsize)
axs[2].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[2].grid(True)

# --- Direction du vent ---
axs[3].plot(pred_WS_BIC_1['Direction'], '--r', linewidth=3, label='WS_BIC_1')
axs[3].plot(pred_WS_BIC_2['Direction'], '--b', linewidth=3, label='WS_BIC_2')
axs[3].plot(test_WS_BIC['Direction du vent'],  'k', linewidth=2, label='Test WS_BIC')
axs[3].set_xlabel("Time", fontsize=label_fontsize)
axs[3].set_ylabel("Direction (°)", fontsize=label_fontsize)
axs[3].tick_params(labelsize=tick_fontsize)
axs[3].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[3].grid(True)
plt.tight_layout()
plt.show()


#%%

fig, axs = plt.subplots(2, 1, figsize=(20, 7), sharex=True)

# --- Vitesse du vent ---
axs[0].plot(pred_IML4_RIKI_1['Vitesse'], '--r', linewidth=3, label='IML4_RIKI_1')
axs[0].plot(pred_IML4_RIKI_2['Vitesse'], '--b', linewidth=3, label='IML4_RIKI_2')
axs[0].plot(test_IML4_RIKI['Vitesse du vent'], 'k', linewidth=2, label='Test IML4_RIKI')
axs[0].set_ylabel("Wind Speed (m/s)", fontsize=label_fontsize)
axs[0].tick_params(labelsize=tick_fontsize)
axs[0].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[0].grid(True)

# --- Direction du vent ---
axs[1].plot(pred_IML4_RIKI_1['Direction'], '--r', linewidth=3, label='IML4_RIKI_1')
axs[1].plot(pred_IML4_RIKI_2['Direction'], '--b', linewidth=3, label='IML4_RIKI_2')
axs[1].plot(test_IML4_RIKI['Direction du vent'],  'k', linewidth=2, label='Test IML4_RIKI')
axs[1].set_xlabel("Time", fontsize=label_fontsize)
axs[1].set_ylabel("Direction (°)", fontsize=label_fontsize)
axs[1].tick_params(labelsize=tick_fontsize)
axs[1].legend(fontsize=tick_fontsize, loc = 'upper right')
axs[1].grid(True)

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
plt.scatter(y_test['Vitesse du vent'], y_pred_dt[0], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# coeffs = np.polyfit(y_test['Eastward Wind'], y_pred_dt[0], deg=1)  # deg=1 pour une droite
# poly_eq = np.poly1d(coeffs)
# plt.plot(y_test['Eastward Wind'], poly_eq(y_test['Eastward Wind']), color='red', label=f"Régression : y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")


# Time series plot of actual vs predicted
plt.subplot(3, 2, 3)
plt.plot(y_pred_dt.index, y_test['Vitesse du vent'], label='Valeurs Réelles')
plt.plot(y_pred_dt.index, y_pred_dt[0], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 5)
errors = y_pred_dt[0].values - y_test['Vitesse du vent'].values
sns.histplot(errors, kde=True, bins=20)
plt.xlabel('Erreur', fontsize=label_fontsize)
plt.ylabel('Fréquence', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Scatter plot of actual vs predicted (Northward Wind)
plt.subplot(3, 2, 2)
plt.scatter(y_test['Direction du vent'], y_pred_dt[1], alpha=0.5)
plt.xlabel('Valeurs Réelles', fontsize=label_fontsize)
plt.ylabel('Prédictions', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# coeffs = np.polyfit(y_test['Northward Wind'], y_pred_dt[1], deg=1)  # deg=1 pour une droite
# poly_eq = np.poly1d(coeffs)
# plt.plot(y_test['Northward Wind'], poly_eq(y_test['Northward Wind']), color='red', label=f"Régression : y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")

# Time series plot of actual vs predicted
plt.subplot(3, 2, 4)
plt.plot(y_pred_dt.index, y_test['Direction du vent'], label='Valeurs Réelles')
plt.plot(y_pred_dt.index, y_pred_dt[1], label='Prédictions', linestyle='--')
plt.xlabel('Temps', fontsize=label_fontsize)
plt.ylabel('Valeur', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.legend(fontsize=legend_fontsize)

# Distribution plot of prediction errors
plt.subplot(3, 2, 6)
errors = y_pred_dt[1].values - y_test['Direction du vent'].values
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

#%%

WS_BIC_test = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/test_2017_ws_bic_v1.csv")
IML4_RIKI_test = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/test_2017_v1.csv")
WS_BIC_test.set_index('Unnamed: 0', inplace = True)
IML4_RIKI_test.set_index('Unnamed: 0', inplace = True)

WS_BIC_1 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_WS_BIC_1_v1.csv")
WS_BIC_2 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_WS_BIC_2_v1.csv")
WS_BIC_1.set_index('Unnamed: 0', inplace = True)
WS_BIC_1.columns = ['Vitesse', 'Direction']
WS_BIC_2.set_index('date', inplace = True)
WS_BIC_2.columns = ['Vitesse', 'Direction']

IML4_RIKI_1 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_IML4_RIKI_1_v1.csv")
IML4_RIKI_2 = pd.read_csv("C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/pred_IML4_RIKI_2_v1.csv")
IML4_RIKI_1.set_index('Unnamed: 0', inplace = True)
IML4_RIKI_1.columns = ['Vitesse', 'Direction']
IML4_RIKI_2.set_index('date', inplace = True)
IML4_RIKI_2.columns = ['Vitesse', 'Direction']

plt.figure(figsize=(18, 15))
plt.scatter(WS_BIC_1.index, WS_BIC_1['Vitesse'])
plt.scatter(WS_BIC_2.index, WS_BIC_2['Vitesse'])
plt.scatter(IML4_RIKI_1.index, IML4_RIKI_1['Vitesse'])
plt.scatter(IML4_RIKI_2.index, IML4_RIKI_2['Vitesse'])
plt.scatter(WS_BIC_test.index, WS_BIC_test['Vitesse du vent'])
plt.scatter(IML4_RIKI_test.index, IML4_RIKI_test['Vitesse du vent'])
plt.xlabel("Importance", fontsize=label_fontsize)
plt.ylabel("Feature", fontsize=label_fontsize)
plt.title("Feature Importance Plot", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()

#%%

date_h0 = '2017-05-10 01:00:00+00:00'
date_h24 = '2017-05-11 01:00:00+00:00'
date_h25 = '2017-05-11 02:00:00+00:00'

df_merged_IML4_RIKI_2_base_n = df_merged_IML4_RIKI_2[['matrice_peaks_tab_n', 'matrice_peaks_tab_p', 'matrice_val_peak_n',
                                                           'matrice_val_peak_p', 'matrice_peaks_freq_n', 'matrice_peaks_freq_p',
                                                           'matrice_offset_n', 'matrice_offset_p', 'matrice_ur', 'Direction du vent', 'Vitesse du vent']]

df_merged_IML4_RIKI_2_base_n.loc[date_h25:, ['Direction du vent', 'Vitesse du vent']] = np.nan

def create_temporal_features3(df, df2, columns, time_windows=['2h', '6h', '12h', '24h']):
    df_copy = df2.copy() # df_window_2
    df = df.copy() # df_window
    # time_windows=['2H']
    reference_time = df_copy.index[0]  # Heure cible, supposée unique
    
    for column in target_columns:
        df_copy[f'{column}_last'] = df[column].iloc[-1]
    
        for window in time_windows:  # par exemple ['2h', '6h', '12h', '24h']
            delta = pd.to_timedelta(window) + pd.to_timedelta('1h')
            mask = (df.index > reference_time - delta) & (df.index <= reference_time)
            subset = df.loc[mask, column]
    
            df_copy[f'{column}_mean_{window}'] = subset.mean()
            df_copy[f'{column}_std_{window}'] = subset.std()
            
    return df_copy.drop(columns=['Vitesse du vent', 'Direction du vent'])

target_columns = ['Vitesse du vent', 'Direction du vent']

from datetime import timedelta
df_merged_IML4_RIKI_2_base = df_merged_IML4_RIKI_2_base_n.copy()
data_train = df_merged_IML4_RIKI_2_base_n.copy()

df_merged_IML4_RIKI_2_base = df_merged_IML4_RIKI_2_base[~df_merged_IML4_RIKI_2_base.index.duplicated()]  # Optionnel : les retirer
df_merged_IML4_RIKI_2_base_n = df_merged_IML4_RIKI_2_base_n[~df_merged_IML4_RIKI_2_base_n.index.duplicated()]
for i in tqdm(range(len(df_merged_IML4_RIKI_2_base))):
    time = df_merged_IML4_RIKI_2_base.index
    time_1 = time[i+47]
    time_2 = time[i+48]
    # Vérifie que time_1 est dans l'index de ton DataFrame original
    if time_1 not in df_merged_IML4_RIKI_2_base.index:
        continue  # saute cette itération

    df_window = df_merged_IML4_RIKI_2_base.loc[time[i]:time_1,['Vitesse du vent', 'Direction du vent']]
    df_window_2 = df_merged_IML4_RIKI_2_base.loc[time_1].to_frame().T
    df_window = df_window.dropna()
    data = create_temporal_features3(df_window, df_window_2, target_columns)

    y_pred2 = model_IML4_RIKI_2.predict(data)  # met dans une liste pour garder la forme (2D)
    df_merged_IML4_RIKI_2_base.loc[time_2, ['Vitesse du vent', 'Direction du vent']] = y_pred2

data_test = df_merged_IML4_RIKI_2[~df_merged_IML4_RIKI_2.index.duplicated()]
df_merged_IML4_RIKI_2_base

plt.figure()

plt.plot(data_test['Vitesse du vent'])
plt.plot(data_train['Vitesse du vent'])
plt.plot(df_merged_IML4_RIKI_2_base['Vitesse du vent'])