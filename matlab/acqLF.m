tic
nameconvention = '21-10-21RamanAN';
pathname = (['C:/temp/' nameconvention '/']);
mkdir(pathname);

acquisitions = 1;
repacktime = 10; 
scantime = 10;



for i = 1:acquisitions
    N = num2str(i);
    fprintf('%d second repack pause before scan %s.\n',repacktime,N)
    pause(repacktime-1)
    fprintf('Beginning scan %s.\n',N)
        tic;
        t = timer;
        t.TimerFcn = @(~,thisEvent)disp(round(toc));
        t.Period = 1;
        t.TasksToExecute = scantime+1;
        t.ExecutionMode = 'fixedRate';
        start(t)
    [data, wavelength] = lfi.acquire();
    writematrix(data,[pathname nameconvention N '.csv'])
    fprintf('Scan %s completed.\n',N)
end

writematrix(wavelength,[pathname nameconvention 'Wavelengths.csv'])
disp('Collection Finished')

beep
pause(1)
beep
pause(1)
beep
toc
