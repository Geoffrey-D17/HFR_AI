function trueDir = math2true(mathDir)

% converts angles (degrees) from mathematical convention
% to true geographic convention
%
% See true2math

% Cedric Chavanne, updated 07/18/2008

trueDir = 90 - mathDir;
trueDir = mod(trueDir,360);
