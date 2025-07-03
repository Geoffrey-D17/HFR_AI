# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:43:08 2025

@author: geofd
"""

from Analyse.Def_et_biblio import *

#%%

data_path = Path("Data/Points_PAB/")

df_X_list = [
    pd.read_csv(data_path / f"point_courant_{i}_radar_PAB_2013.csv")
    for i in range(1, 16)]

df_y = obtenir_variables_fichiers_netCDF("Data/Data_ref/wave/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1743014497245.nc")

lon, lat = np.meshgrid(df_y['longitude'], df_y['latitude'])
lon_p = [df['lon'][0] for df in df_X_list]
lat_p = [df['lat'][0] for df in df_X_list]
X_wave = pd.DataFrame(df_y['VSDX'][:,5,6], columns = ['VSDX'])
Y_wave =  pd.DataFrame(df_y['VSDY'][:,5,6], columns = ['VSDY'])
time = pd.DataFrame(pd.to_datetime(df_y['time'], unit = 's'), columns = ['time'])
df_yobs = pd.concat([X_wave, Y_wave, time], axis = 1)

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)

ax.scatter(lon[5,6], lat[5,6], color='red', marker='o', label="Radar PAB", transform=ccrs.PlateCarree())
ax.scatter(lon_p, lat_p, color='blue', marker='o', label="Radar PAB", transform=ccrs.PlateCarree())
ax.scatter( crad_lon, crad_lat)#lon_c[3,3:6], lat_c[3,3:6])
plt.title("Direction du vent estimée par Radar HF")
plt.legend()
plt.show()

target_columns = ['VSDX', 'VSDY']

x_train_list, y_train_list, x_test_list, y_test_list = zip(*[prepare_time_series(df_X, df_yobs, target_columns) for df_X in df_X_list])

x_train, y_train = pd.concat(x_train_list, axis = 0), pd.concat(y_train_list, axis = 0)
x_test, y_test = pd.concat(x_test_list, axis = 0), pd.concat(y_test_list, axis = 0)

x_train, y_train = x_train.to_numpy().astype(float), y_train.to_numpy().astype(float)
x_test, y_test = x_test.to_numpy().astype(float), y_test.to_numpy().astype(float)


#%%
model_Regressor = RandomForestRegressor(n_estimators=10, criterion='absolute_error')
model_Regressor.fit(x_train, y_train)
y_pred_Regressor = model_Regressor.predict(x_test)
mae_Regressor = mean_absolute_error(y_test, y_pred_Regressor)
# rmse = root_mean_squared_error(y_test, y_pred)
print(f"MAE Regressor: {mae_Regressor:.3f}")

y_test_df = pd.DataFrame(y_test)
y_pred_Regressor_df = pd.DataFrame(y_pred_Regressor)
# y_pred_Embedding_df = pd.DataFrame(y_pred_Embedding)

plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.plot(y_test_df[0], color='blue', label='Valeurs Réelles ue_0m', linewidth=2)
plt.plot(y_pred_Regressor_df[0], ':', color='red', alpha=0.7, label='Prédictions ue_0m', linewidth=2)

plt.title("Comparaison des Valeurs Réelles et Prédictions - ue_0m", fontsize=15)
plt.xlabel("Temps", fontsize=15)
plt.ylabel("Valeur", fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(y_test_df[1], color='blue', label='Valeurs Réelles ve_0m', linewidth=2)
plt.plot(y_pred_Regressor_df[1], ':', color='red', alpha=0.7, label='Prédictions ve_0m', linewidth=2)

plt.title("Comparaison des Valeurs Réelles et Prédictions - ve_0m", fontsize=15)
plt.xlabel("Temps", fontsize=15)
plt.ylabel("Valeur", fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()