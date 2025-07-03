
input_folder = 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\spec';
output_folder= 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\spec_\';
file_list = dir(fullfile(input_folder, '*spec')); % .crad_pol_lst

% Boucle sur tous les fichiers trouvés
for i = 1:length(file_list)
    file_name = file_list(i).name;
    file_path = fullfile(input_folder, file_name);
    
    % Extraction des données
    [Time, LON, LAT, X, Y, freq, fbragg, PXY] = read_WERA_spec(file_path);

    [~, base_name, ~] = fileparts(file_name);
    
    % Sauvegarde des variables dans un fichier .mat
    save(fullfile(output_folder, [base_name 'PXY.mat']), 'PXY');
    
    fprintf('Fichier traité : %s\n', file_name);
end
