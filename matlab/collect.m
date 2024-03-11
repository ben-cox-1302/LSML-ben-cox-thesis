mode = "Raman";
lambda = 355;
oldoffset = 0;%.242684;
nameconvention = 'GlassG60';
pathname = 'C:/temp/';
y1 = 1;
y2 = 253;

collections = 1;
M = zeros(253,1024,collections);

for i = 1:collections
    M(:,:,i) = readmatrix([pathname nameconvention '/' nameconvention num2str(i) '.csv']);
end

MWavelengths = readmatrix([pathname nameconvention '/' nameconvention 'Wavelengths.csv']);
if lambda == 355
    wavenumber_script_355;
elseif lambda == 532
    wavenumber_script_532;
elseif lambda == 1064
    wavenumber_script_1064;
else
    disp('Incorrect Wavelength Entered')
end
MWavenumbers = RWavenumber-(1./((MWavelengths+offset+oldoffset)*nm_to_m)./100);

if mode == "Raman"
    hold on
for i = 1:collections
    N = num2str(i);
    plot(MWavenumbers,mean(M(y1:y2,:,i)))
    writematrix(mean(M(:,:,i)).',[pathname nameconvention '/' nameconvention 'Average' N '.csv'])
    Get = readmatrix([pathname nameconvention '/' nameconvention 'Average' N '.csv']);
    Combine = [MWavenumbers Get];
    writematrix(Combine,[pathname nameconvention '/' nameconvention 'Average' N '.csv'])
end
hold off
    
xlabel('Wavenumber (cm^{-1})')
ylabel('Intensity (counts)')
xlim([000 1500]);
legend({'Light Off','3 ns','30 ns','300 ns','3 us', '30 us',},'Location','northwest')

elseif mode == "LIBS"
         hold on
for i = 1:collections
    N = num2str(i);
    plot(MWavelengths+offset,mean(M(:,:,i)))
end
hold off
    else
            disp('Incorrect Mode Entered')
end
    