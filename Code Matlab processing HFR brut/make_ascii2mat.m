%% PAO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all
% t1 = datenum(2013,1,1);
% t2 = datenum(2020,12,31);
% theta1=280-90-60;
% theta2=280-90+60;
% load 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat'
% slon = xne;
% slat = yne;
% 
% T = t1:t2;
% error_files=[];
% saved_files = {};  % au début du script
% 
% for i = 1:length(T)
%     date = datevec(T(i));
%     year = date(1);
%     jday = julian(T(i));
%     D = dir(['D:\Data_HFR\PAO\crad_pol_ascii\','*ascii']);
% %     folder = fullfile('C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_ascii\', [num2str(year), num2str(jday,'%03i')]);
% %     D = dir(fullfile(folder, '*.crad_pol_ascii'));
%     n = length(D);
%     if n>0
% %         for j = 1:n
% %             name = D(j).name;
% %             filein = ['C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_ascii\',name];
% %             fileout=[filein(1:end-15),'.mat'];
% %             hour = str2num(name(8:9));
% %             minute = str2num(name(10:11));
% %             date(4) = hour;
% %             date(5) = minute;
% %             t = datenum(date);
% %             tzone = 'UTC';
% %             try
% %                 [r,theta,X,Y,Ur,Ustd,SN,Pow,Nb,S,N]=ascii2mat(filein,theta1,theta2,slon,slat);
% %                 save(fileout,'Ur','Ustd','SN','Pow','Nb','S','N','X','Y','r','theta','t','tzone');
% %             catch
% %                 error_files=[error_files;filein];
% %             end
%         for j = 1:n
%             name = D(j).name;
% %             filein = ['D:\Data_HFR\PAO\CRAD_mat2\', name];
% %             filein = fullfile(input_dir, D(k).name);
%             disp(D(i).name)  % pour vérifier
%             filein = fullfile(D(j).folder, D(j).name);
% 
%             % --> Nouvelle façon de générer le fichier .mat dans le même dossier
%             [filefolder, name_no_ext, ~] = fileparts(filein);
%             fileout = fullfile('D:\Data_HFR\PAO\CRAD_mat2', [name_no_ext, '_crad.mat']);
% 
%             hour = str2num(name(8:9));
%             minute = str2num(name(10:11));
%             date(4) = hour;
%             date(5) = minute;
%             t = datenum(date);
%             tzone = 'UTC';
% 
%             try
%                 [r, theta, X, Y, Ur, Ustd, SN, Pow, Nb, S, N] = ascii2mat(filein, theta1, theta2, slon, slat);
%                 save(fileout, 'Ur', 'Ustd', 'SN', 'Pow', 'Nb', 'S', 'N', 'X', 'Y', 'r', 'theta', 't', 'tzone');
% 
%                 % Dans la boucle, après chaque sauvegarde réussie :
%                 saved_files{end+1} = fileout;
% 
%                 % À la fin du script :
%                 disp('--- Fichiers .mat sauvegardés ---');
%                 for k = 1:length(saved_files)
%                     disp(saved_files{k});
%                 end
% 
%             catch
%                 error_files = [error_files; filein];
%             end
%         end
%     end
%     fprintf('%s done\n',datestr(T(i),1));
% end

% %% PAB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
clear all
t1 = datenum(2013,1,1);
t2 = datenum(2020,12,31);
theta1=205-90-70;
theta2=205-90+70;
load 'C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAB\sites.mat'
slon = xnw;
slat = ynw;

T = t1:t2;
error_files=[];
% 
% for i = 1:length(T)
%     date = datevec(T(i));
%     year = date(1);
%     jday = julian(T(i));
%     D = dir(['../../../../data/st-laurent/WERAs/PAB/radials/',num2str(year),num2str(jday,'%03i'),'*.crad_pol_ascii']);
%     n = length(D);
%     if n>0
%         for j = 1:n
%             name = D(j).name;
%             filein = ['../../../../data/st-laurent/WERAs/PAB/radials/',name];
%             fileout=[filein(1:end-15),'.mat'];
%             hour = str2num(name(8:9));
%             minute = str2num(name(10:11));
%             date(4) = hour;
%             date(5) = minute;
%             t = datenum(date);
%             tzone = 'UTC';
%             try
%                 [r,theta,X,Y,Ur,Ustd,SN,Pow,Nb,S,N]=ascii2mat(filein,theta1,theta2,slon,slat);
%                 save(fileout,'Ur','Ustd','SN','Pow','Nb','S','N','X','Y','r','theta','t','tzone');
%             catch
%                 error_files=[error_files;filein];
%             end
%         end
%     end
%     fprintf('%s done\n',datestr(T(i),1));
% end

for i = 1:length(T)
    date = datevec(T(i));
    year = date(1);
    jday = julian(T(i));
    D = dir(['D:\Data_HFR\PAB\crad_pol_ascii\','*ascii']);
%     folder = fullfile('C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_ascii\', [num2str(year), num2str(jday,'%03i')]);
%     D = dir(fullfile(folder, '*.crad_pol_ascii'));
    n = length(D);
    if n>0
%         for j = 1:n
%             name = D(j).name;
%             filein = ['C:\Users\geofd\OneDrive\Documents\Stage_ISMER\Data\Radar HF\WERA\PAO\crad_ascii\',name];
%             fileout=[filein(1:end-15),'.mat'];
%             hour = str2num(name(8:9));
%             minute = str2num(name(10:11));
%             date(4) = hour;
%             date(5) = minute;
%             t = datenum(date);
%             tzone = 'UTC';
%             try
%                 [r,theta,X,Y,Ur,Ustd,SN,Pow,Nb,S,N]=ascii2mat(filein,theta1,theta2,slon,slat);
%                 save(fileout,'Ur','Ustd','SN','Pow','Nb','S','N','X','Y','r','theta','t','tzone');
%             catch
%                 error_files=[error_files;filein];
%             end
        for j = 1:n
            name = D(i).name;
%             filein = ['D:\Data_HFR\PAO\CRAD_mat2\', name];
%             filein = fullfile(input_dir, D(k).name);
            disp(D(1).name)  % pour vérifier
            filein = fullfile(D(j).folder, D(j).name);

            % --> Nouvelle façon de générer le fichier .mat dans le même dossier
            [filefolder, name_no_ext, ~] = fileparts(filein);
            fileout = fullfile('D:\Data_HFR\PAB\CRAD_mat2', [name_no_ext, '_crad.mat']);

            hour = str2num(name(8:9));
            minute = str2num(name(10:11));
            date(4) = hour;
            date(5) = minute;
            t = datenum(date);
            tzone = 'UTC';

            try
                [r, theta, X, Y, Ur, Ustd, SN, Pow, Nb, S, N] = ascii2mat(filein, theta1, theta2, slon, slat);
                save(fileout, 'Ur', 'Ustd', 'SN', 'Pow', 'Nb', 'S', 'N', 'X', 'Y', 'r', 'theta', 't', 'tzone');

                % Dans la boucle, après chaque sauvegarde réussie :
                saved_files{end+1} = fileout;

                % À la fin du script :
                disp('--- Fichiers .mat sauvegardés ---');
                for k = 1:length(saved_files)
                    disp(saved_files{k});
                end

            catch
                error_files = [error_files; filein];
            end
        end
    end
    fprintf('%s done\n',datestr(T(i),1));
end
