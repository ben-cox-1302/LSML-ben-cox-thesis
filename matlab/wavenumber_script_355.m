Rayleigh_collect_script_355;
averages = 1;
nm_to_m = 10^(-9);
RWavenumbersAll = zeros(1,averages);
clear RWavenumberAvg
clear RWavenumber
clear offset

for i = 1:averages
    Ri = mean(R(:,:,i));
    [maxYValue, indexAtMaxY] = max(Ri);
    xValueAtMaxYValue = RWavelengths(indexAtMaxY);
    RWavenumbersAll(:,i) = (1/(xValueAtMaxYValue.*nm_to_m)/100);
end
RWavenumberAvg = mean(RWavenumbersAll);
offset = 355-((1/(100*RWavenumberAvg))./nm_to_m);
RWavenumber = 1/(((1/(RWavenumberAvg*100))/nm_to_m + offset)*nm_to_m)/100;