%% Setting Up the Environment for LightField

automation_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightField.AutomationV3.dll';
addin_path = 'C:\Program Files\Princeton Instruments\LightField\AddInViews\PrincetonInstruments.LightFieldViewV3.dll';
support_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightFieldAddInSupportServices.dll';
    
addin_class = NET.addAssembly(addin_path);
automation_class = NET.addAssembly(automation_path);
support_class = NET.addAssembly(support_path);
import PrincetonInstruments.LightField.AddIns.*;

%% Setting the Defaults for the Experiment

% Makes the LightField Software UI Visible
lfi = lfm(true);

% Assuming this preloads default settings and environment setup?
lfi.load_experiment('Raman10m');

disp('Ready for experiments (pending camera cooldown)')

%% Setup Path Conventions

tic
nameconvention = '11-10-23-NdYAGPos7Temp450time100-120';
pathname = (['C:/temp/' nameconvention '/']);
mkdir(pathname);

%% Variables to Change during Experiment Execution

acquisitions = 1;
repacktime = 1;     
scantime = 1;

% Ideas:
% increase number of acquisitions
% Change the time period 
% Change the execution mode

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