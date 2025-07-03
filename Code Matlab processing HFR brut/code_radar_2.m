% file = "C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA_2\20130010040_pab.crad_pol_lst";
% [Ur,Acc,Var,Pow,X,Y,r,theta,t,tzone] = lst2mat(file, 45, 185, -68.4519, 49.0422);
% 
% save_file = 'C:/Users/geofd/OneDrive/Documents/Stage_ISMER/Data/Radar HF/WERA/PAB/CRAD/';
% 
% writematrix(Ur, fullfile(save_file, 'Ur_20130010040_pab.csv'));
% writematrix(Acc, fullfile(save_file, 'Acc_20130010040_pab.csv'));
% writematrix(Var, fullfile(save_file, 'Var_20130010040_pab.csv'));
% writematrix(Pow, fullfile(save_file, 'Pow_20130010040_pab.csv'));
% writematrix(X, fullfile(save_file, 'X_20130010040_pab.csv'));
% writematrix(Y, fullfile(save_file, 'Y_20130010040_pab.csv'));
% writematrix(r, fullfile(save_file, 'r_20130010040_pab.csv'));
% writematrix(theta, fullfile(save_file, 'theta_20130010040_pab.csv'));
% writematrix(t, fullfile(save_file, 't_20130010040_pab.csv'));

% writetable(tzone, save_file+'*tzone_20130010040_pab.csv');

% % Définition des dossiers
% output_folder = 'D:\HFR\PAB\2013\';
% input_folder = 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\CRAD_2\';
% 
% Liste tous les fichiers .crad_pol_lst dans le dossier d'entrée
input_folder = 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_ascii\converted_lst2';
output_folder= 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_\';
file_list = dir(fullfile(input_folder, '*lst')); % .crad_pol_lst

% Coordonnées et paramètres de lst2mat (à ajuster si nécessaire)
lat_min = 130; %45;
lat_max = 250; %185;
lon_ref = -68.457995752950010; %-69.250000; %-69.1342;
lat_ref = 49.042624566650005; %49.033333; %48.5733;
site = load ("C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat");
% Boucle sur tous les fichiers trouvés
for i = 1:length(file_list)
    file_name = file_list(i).name;
    file_path = fullfile(input_folder, file_name);
    
    % Extraction des données
    [Ur, Acc, Var, Pow, X, Y, r, theta, t, tzone] = lst2mat(file_path, lat_min, lat_max, lon_ref, lat_ref);
    [Theta,R]=meshgrid(true2math(theta),r);
    [X,Y]=ra2lonlat(R,Theta,site.xne,site.yne);

    [~, base_name, ~] = fileparts(file_name);
    % Sauvegarde des variables en CSV
    writematrix(Ur, fullfile(output_folder, [base_name '_Ur.csv']));
    writematrix(X, fullfile(output_folder, [base_name '_X.csv']));
    writematrix(Y, fullfile(output_folder, [base_name '_Y.csv']));
    
    fprintf('Fichier traité : %s\n', file_name);
end
% 
% input_folder = 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\crad\';
% output_folder= 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\';

% % Vérifier si le dossier de sortie existe, sinon le créer
% if ~exist(output_folder, 'dir')
%     mkdir(output_folder);
% end
% 
% % Trouver tous les sous-dossiers dans input_folder
% subfolders = dir(input_folder);
% subfolders = subfolders([subfolders.isdir]); % Garder seulement les dossiers
% subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Exclure . et ..
% 
% % Liste de tous les fichiers .crad_pol_lst dans tous les sous-dossiers
% file_list_2 = [];
% for k = 1:length(subfolders)
%     subfolder_path = fullfile(input_folder, subfolders(k).name);
%     files_in_subfolder = dir(fullfile(subfolder_path, '*.crad_pol_lst'));
%     for j = 1:length(files_in_subfolder)
%         files_in_subfolder(j).folder = subfolder_path; % Ajouter le chemin complet
%     end
%     file_list_2 = [file_list_2; files_in_subfolder]; % Concaténer la liste des fichiers
% end
% 
% % Vérification s'il y a des fichiers
% if isempty(file_list_2)
%     error('Aucun fichier .crad_pol_lst trouvé dans les sous-dossiers de %s', input_folder);
% end
% 
% % Coordonnées et paramètres de lst2mat (à ajuster si nécessaire)
% lat_min = 45;
% lat_max = 185;
% lon_ref = -69.1342;
% lat_ref = 48.5733;
% 
% % Charger les données de site
% site_file = "C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat";
% if exist(site_file, 'file')
%     load(site_file);
% else
%     error('Fichier sites.mat introuvable !');
% end
% 
% % Boucle sur tous les fichiers trouvés
% for i = 1:length(file_list_2)
%     file_name = file_list_2(i).name;
%     file_path = fullfile(file_list_2(i).folder, file_name);
%     
%     try
%         % Extraction des données avec lst2mat
%         [Ur, Acc, Var, Pow, X, Y, r, theta, t, tzone] = lst2mat(file_path, lat_min, lat_max, lon_ref, lat_ref);
% 
%         % Vérification si les variables sont bien retournées
%         if isempty(Ur) || isempty(X) || isempty(Y)
%             warning('Données vides pour %s, passage au fichier suivant...', file_name);
%             continue;
%         end
%         
%         % Conversion des coordonnées
%         [Theta, R] = meshgrid(true2math(theta), r);
%         [X, Y] = ra2lonlat(R, Theta, xnw, ynw);
% 
%         % Construction du nom de base sans extension
%         [~, base_name, ~] = fileparts(file_name);
% 
%         % Sauvegarde des variables en CSV
%         writematrix(Ur, fullfile(output_folder, [base_name '_Ur.csv']));
%         writematrix(X, fullfile(output_folder, [base_name '_X.csv']));
%         writematrix(Y, fullfile(output_folder, [base_name '_Y.csv']));
%         
%         fprintf('✅ Fichier traité avec succès : %s\n', file_name);
% 
%     catch ME
%         warning('❌ Erreur lors du traitement du fichier %s : %s', file_name, ME.message);
%         continue;
%     end
% end

