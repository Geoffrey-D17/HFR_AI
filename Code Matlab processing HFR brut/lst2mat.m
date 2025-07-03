function [Ur,Acc,Var,Pow,X,Y,r,theta,t,tzone]=lst2mat(file,a1,a2,slon,slat)

% [Ur,Acc,Var,Pow,X,Y,r,theta,t,tzone]=lst2mat(file,a1,a2);
%
% load the ascii data in *_lst file into matlab matrices:
% Ur (radial current in m/s)
% Acc (radial current accuracy in m/s)
% Var (radial current standard deviation in m/s)
% Pow (relative backscatter power)
% on the polar grid with geographic coordinates (X,Y),
% corresponding to distances from the radar r (in km) and
% azimuth angles theta (in geographic degrees), for azimuths
% comprised between a1 and a2 (in geographic degrees and clockwise).
% Also returns the time t (in datenum format) and time zone
% tzone, based on the header of the file.

% Cedric Chavanne, updated 07/18/2008

fic=fopen(file,'r');
if fic<0
  fprintf('cannot open the file %s\n',file)
  return
end

% extract information from header line:

ln1 = fgetl(fic);
if ~isstr(ln1)
   fprintf('wrong file format for %s\n',file)
   return
end
t=datenum(datestr(ln1(1:18),0));
tzone=ln1(20:22);
% slat = str2num(ln1(38:47));
% slon = str2num(ln1(53:61));
ns   = ln1(49);
if isequal(ns,'S')
   slat = -slat;
end
ew   = ln1(63);
if isequal(ew,'W')
   slon = -slon;
end

% count number of range cells:

i=0;
while 1
   i=i+1;
   ln3 = fgetl(fic);
   if ~isstr(ln3)
      break
   end
%   r(i) = str2num(ln2(13:20));
   nrow = str2num(ln3(35:end));
   if nrow > 0
     ln4 = fgetl(fic);
     if ~isstr(ln4)
         break
     end
     M=str2num(fscanf(fic,'%c',[61,nrow])');
%     if size(M,2)==5
%       Ur(i,M(:,1))=M(:,2)';
%       Acc(i,M(:,1))=M(:,3)';
%       Var(i,M(:,1))=M(:,4)';
%       Pow(i,M(:,1))=M(:,5)';
%     end
   end
end

fclose(fic);

% initialize matrices:

r=NaN*ones(i,1);
theta=1:360;
Ur=NaN*ones(i,360);
Acc=NaN*ones(i,360);
Var=NaN*ones(i,360);
Pow=NaN*ones(i,360);

%start loop over range cells

fic=fopen(file,'r');
ln1 = fgetl(fic);
i=0;
while 1
   i=i+1;
   ln2 = fgetl(fic);
   ln3 = fgetl(fic);
   if ~isstr(ln2)
      break
   end
   r(i) = str2num(ln2(11:end));
   nrow = str2num(ln3(35:end));
   if nrow > 0
     ln4 = fgetl(fic);
     if ~isstr(ln4)
         break
     end
     M=str2num(fscanf(fic,'%c',[61,nrow])');
     if size(M,2)==5
       Ur(i,M(:,1))=M(:,2)';
       Acc(i,M(:,1))=M(:,3)';
       Var(i,M(:,1))=M(:,4)';
       Pow(i,M(:,1))=M(:,5)';
     end
   end
end

fclose(fic);

% clip unused azimuths:
if (a1 <= a2)
  I=find(theta>=a1 & theta<=a2);
else
  I=[find(theta>=a1) find(theta<=a2)];
end
Ur=Ur(:,I);
Acc=Acc(:,I);
Var=Var(:,I);
Pow=Pow(:,I);
theta=theta(I);

% make the grid coordinates:
[Theta,R]=meshgrid(true2math(theta),r);
[X,Y]=ra2lonlat(R,Theta,slon,slat);
