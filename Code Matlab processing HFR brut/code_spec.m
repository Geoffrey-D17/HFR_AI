clear all; close all; clc;
% 
% % Période d'intérêt
% t1 = datenum(2013,1,1);
% t2 = datenum(2020,12,31);
% T = t1:t2;
% 
% % Chargement du site (coordonnées radar)
% load('C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat');
% slon = xne;  % longitudes station
% slat = yne;  % latitudes station
% 
% % Paramètres angulaires
% theta1 = 280 - 90 - 60;
% theta2 = 280 - 90 + 60;

t1 = datenum(2013,1,1);
t2 = datenum(2020,12,31);
theta1=205-90-70;
theta2=205-90+70;
load 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat'
slon = xnw;
slat = ynw;

T = t1:t2;
error_files=[];

% Dossier contenant les fichiers .spec
input_dir = 'D:\Data_HFR\PAB\spec\'; % à adapter
output_dir = 'D:\Data_HFR\PAB\SPEC_mat';  % peut être changé

saved_files = {};
error_files = {};

for i = 1:length(T)
    date = datevec(T(i));
    year = date(1);
    jday = julian(T(i));

    % Lister les fichiers de ce jour
    D1 = dir(fullfile(input_dir, '2019*'));
    D2 = dir(fullfile(input_dir, '2020*'));
    D = [D1; D2];
    n = length(D);

    if n > 0
        for j = 1:n
            name = D(j).name;
            filein = fullfile(input_dir, name);

            % Construction du fichier de sortie
            [~, name_no_ext, ~] = fileparts(name);
            fileout = fullfile(output_dir, [name_no_ext, '_spec.mat']);

            try
                % Lecture des données WERA
                [Time, LON, LAT, X, Y, freq, fbragg, PXY] = read_WERA_spec(filein);

                % Sauvegarde des variables utiles
                save(fileout, 'Time', 'LON', 'LAT', 'X', 'Y', 'freq', 'fbragg', 'PXY');
                saved_files{end+1} = fileout;

            catch ME
                warning('Erreur sur %s : %s', filein, ME.message);
                error_files{end+1} = filein;
            end
        end
    end

    fprintf('%s done\n', datestr(T(i), 1));
end

% Affichage des fichiers traités
disp('--- Fichiers .mat sauvegardés ---');
for k = 1:length(saved_files)
    disp(saved_files{k});
end

if ~isempty(error_files)
    disp('--- Fichiers en erreur ---');
    for k = 1:length(error_files)
        disp(error_files{k});
    end
end
