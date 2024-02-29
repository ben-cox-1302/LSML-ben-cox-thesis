 automation_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightField.AutomationV3.dll';
addin_path = 'C:\Program Files\Princeton Instruments\LightField\AddInViews\PrincetonInstruments.LightFieldViewV3.dll';
support_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightFieldAddInSupportServices.dll';
    
addin_class = NET.addAssembly(addin_path);
automation_class = NET.addAssembly(automation_path);
support_class = NET.addAssembly(support_path);
import PrincetonInstruments.LightField.AddIns.*;              

out.addinbase = PrincetonInstruments.LightField.AddIns.AddInBase();
            out.automation = PrincetonInstruments.LightField.Automation.Automation(true,[]);
            out.application = out.automation.LightFieldApplication;
            out.experiment = out.application.Experiment;       
out.experiment.Load('Raman10m');


import System.IO.FileAccess;
            out.experiment.Acquire();
            accessed_wavelength = 0;
            
            
while out.experiment.IsRunning % During acquisition...
                % Case where wavelength is empty
                if accessed_wavelength == 0 && isempty(out.experiment.SystemColumnCalibration)
                    fprintf('Wavelength information not available\n');
                    wavelength = [];
                    accessed_wavelength = 1;
                elseif accessed_wavelength == 0
                    wavelen_len = out.experiment.SystemColumnCalibration.Length;
                    assert(wavelen_len >= 1);
                    wavelength = zeros(wavelen_len, 1);
                    for i = 0:wavelen_len-1 % load wavelength info directly from LightField instance
                        wavelength(i+1) = out.experiment.SystemColumnCalibration.Get(i);
                    end
                    accessed_wavelength = 1;
                end
end

lastfile = out.application.FileManager.GetRecentlyAcquiredFileNames.GetItem(0);
imageset = out.application.FileManager.OpenFile(lastfile,FileAccess.Read);
            
            if imageset.Regions.Length == 1
                if imageset.Frames == 1
                    frame = imageset.GetFrame(0,0);
                    data = reshape(frame.GetData().double,frame.Width,frame.Height)';
                    return;
                else
                    data = [];
                    for i = 0:imageset.Frames-1
                        frame = imageset.GetFrame(0,i);
                        data = cat(3,data,reshape(frame.GetData().double,frame.Width,frame.Height,1)');
                    end
                    return;
                end
            else
                data = cell(imageset.Regions.Length,1);
                for j = 0:imageset.Regions.Length-1
                    if imageset.Frames == 1
                        frame = imageset.GetFrame(j,0);
                        buffer = reshape(frame.GetData().double,frame.Width,frame.Height)';
                    else
                        buffer = [];
                        for i = 0:imageset.Frames-1
                            frame = imageset.GetFrame(j,i);
                            buffer = cat(3,buffer,reshape(frame.GetData().double,frame.Width,frame.Height,1)');
                        end
                    end
                    data{j+1} = buffer;
                end
            end
        