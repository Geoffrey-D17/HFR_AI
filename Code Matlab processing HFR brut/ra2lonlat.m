function [X,Y]=ra2lonlat(r,a,x0,y0)
% r = R; a = Theta; x0 = slon; y0 = slat;
% [X,Y]=ra2lonlat(r,a,x0,y0)
%
% convert range r (in km) and angle a (in mathematical degrees)
% relative to geographic point (x0,y0)
% into longitude X and latitude Y
% rem: if r and a are vectors with different orientations,
% then 2D matrices are created with meshgrid to compute X and Y.

% Cedric Chavanne, updated 07/18/2008

% check if r and a are vectors of different orientation:
[m1,n1]=size(a);
[m2,n2]=size(r);
if (m1~=m2 | n1~=n2)
  [a,r]=meshgrid(a,r);
end

a=math2true(a);
r=km2deg(r);
[Y,X]=reckon(y0*ones(size(r)),x0*ones(size(r)),r,a);

