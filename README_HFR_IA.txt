For using the AI/HFR project:
======================================================================================================
Data:
  --> PAB/PAO       # All available data for both HFRs from 2013 to 2020 creat  with HFR WERA data with "Code Matlab processing HFR brut" and PAB_data/PAO_data and creation_data_liste
  --> OGSL        # Tide gauge data downlaod in OGSL : https://ogsl.ca/conditions/?lg=fr
  --> data_ref      # Copernicus data (currents, waves...) from initial tests downlaod in Copernicus data store : https://data.marine.copernicus.eu/products
  --> Bouée       # IML4 buoy data downlaod in OGSL : https://ogsl.ca/conditions/?lg=fr , PMZA-RIKI
  --> data_station_météo # Weather station data from BIC downlaod in OGSL : https://ogsl.ca/conditions/?lg=fr 
  --> Point_PAB     # Data used in various test models and ERA5 data for PAB for specific points in PAB/PAO and download in ERA5 data store : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
  --> Point_PAO     # Data used in various test models and ERA5 data for PAO for PAB for specific points in PAB/PAO and download in ERA5 data store : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
------------------------------------------------------------------------------------------------------
LoadData:
  --> Def_et_Biblio   # Most of the definitions and libraries for this project
  --> PAB_data     # Loading PAB data
  --> PAO_data     # Loading PAO data
  --> creation_data_liste   # Creating data files for both HFRs as lists of DataFrames
  --> creation_data_matrice  # Creating data files for both HFRs as 3D matrices (t, x, y)
  --> creation_data_plus_proche_points_liste # Creating data files for both HFRs as lists of the x closest points in DataFrame form. Needs reconfiguration for PAB or PAO
  --> PAB_traitement_model  # Processing PAB data for data creation with creation_data_liste
  --> PAO_traitement_model  # Processing PAO data for data creation with creation_data_liste
------------------------------------------------------------------------------------------------------
Models:
  --> code_RF_radar_hf_vent_in_situ_geoffrey_30_06_2025 # Final models type 1 and 2 using buoy and weather station in-situ data
  --> code_RF_radar_hf_geoffrey_17_03_2025     # Type 1 wind model using ERA5 data (from Point_PAB and Point_PAO)
  --> code_RF_radar_hf_ayoube_01_05_2025     # Type 2 wind model using ERA5 data (from Point_PAB and Point_PAO)
  --> code_RF_radar_hf_vague_geoffrey_26_03_2025  # Type 1 wave model using Copernicus data (from Point_PAB and Point_PAO)
  --> code_RF_radar_hf_courant_geoffrey_25_03_2025 # Type 1 current model using Copernicus data (from Point_PAB and Point_PAO)
  --> code_LSTM_radar_hf        # Type 1 wind model using an LSTM neural network with ERA5 data (from Point_PAB and Point_PAO)
------------------------------------------------------------------------------------------------------
Notes:
The most recent model from 30/06/2025, using in situ data, uses all available data from the HFRs.
Suggestion for improvement: Instead of providing the HFRs with only the amplitudes, frequencies, and offsets of the energy peaks, provide the full spectrum for each point. 
