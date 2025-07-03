function [r,az,X,Y,Ur,Ustd,SN,P,Nb,S,N,t,tzone]=ascii2mat(file,a1,a2,slon,slat)

% [r,az,X,Y,Ur,Ustd,SN,P,Nb,S,N]=ascii2mat(file,a1,a2,slon,slat);

% Cedric Chavanne, updated 05/05/2014
% Marion Bandet, updated number of antennas 11/24/2016

fprintf('=================== Start ascii2mat ==================\n')
fprintf('Treating file %s\n',file)

if exist(file)
    fid = fopen(file,'r');
    ln1 = fgetl(fid);
    if ~isstr(ln1)
        fprintf('wrong file format for %s\n',file)
        return
    end
    t = datenum(datestr(ln1(17:31),0));
    tzone = ln1(33:35);
%     t = datenum(datestr(ln1(1:17),0));
%     tzone = ln1(19:21);

    fclose(fid);
    
    fid=fopen(file,'r');
    header=char(transpose(fread(fid,512,'uchar')));
    nsamp=sscanf(header,'%g');
	i=findstr(header,'ANT:');
	%nant=sscanf(header(i(1)+4:end),'%g');
	i=findstr(header,'RANGE:');
	dr=sscanf(header(i(1)+6:end),'%g');
	i=findstr(header,'NRRANGES:');
	nr=sscanf(header(i(1)+10:i(1)+12),'%g');
    line = fgetl(fid);
    line = fgetl(fid);
    nant = abs(sscanf(line(13:14),'%g'));
    S = NaN(nr,nant);
    N = NaN(nr,nant);
    AZ = 1:360;
    Nb = NaN(nr,360);
    U = NaN(nr,360);
    V = NaN(nr,360);
    SN = NaN(nr,360);
    P = NaN(nr,360);
    for j = 1:nr
        for i = 1:nant
            line = str2num(fgetl(fid));
            S(j,i) = line(1);
            N(j,i) = line(2);
        end
        n = str2num(fgetl(fid));
        if n > 0
            for i = 1:n
                line = str2num(fgetl(fid));
                az = line(1);
                Nb(j,az) = line(2);
                U(j,az) = line(3);
                V(j,az) = line(4);
                SN(j,az) = line(5);
                P(j,az) = line(6);
            end
        end
    end
    fclose(fid);
    
    % clip unused azimuths:
    if (a1 <= a2)
        J=find(AZ>=a1 & AZ<=a2);
    else
        J=[find(AZ>=a1) find(AZ<=a2)];
    end
    Nb = Nb(:,J);
    U = U(:,J);
    V = V(:,J);
    SN = SN(:,J);
    P = P(:,J);
    az = AZ(J);
    
    % make the grid coordinates:
    r = dr:dr:nr*dr;
    [Theta,R]=meshgrid(true2math(az),r);
    [X,Y]=ra2lonlat(R,Theta,slon,slat);
    r = r';
    
    % compute mean radial velocity and standard deviation:
    Ur = U./SN;
    Ustd = sqrt(V./SN - Ur.^2);
else
    disp(['file ',file,' not found!'])
    r = [];
    az = [];
    X = [];
    Y = [];
    Ur = [];
    Ustd = [];
    SN = [];
    P = [];
    Nb = [];
    S = [];
    N = [];
end

fprintf('=================== End ascii2mat ==================\n')
