nameconvention355 = '19-10-21Rayleigh';
pathname = 'C:/temp/';

Rcollections = 1;
R = zeros(253,1024,Rcollections);

for i = 1:Rcollections
    R(:,:,i) = readmatrix([pathname nameconvention355 '/' nameconvention355 num2str(i) '.csv']);
end


RWavelengths = readmatrix([pathname nameconvention355 '/' nameconvention355 'Wavelengths.csv']);
