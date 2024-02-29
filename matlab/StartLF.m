automation_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightField.AutomationV3.dll';
addin_path = 'C:\Program Files\Princeton Instruments\LightField\AddInViews\PrincetonInstruments.LightFieldViewV3.dll';
support_path = 'C:\Program Files\Princeton Instruments\LightField\PrincetonInstruments.LightFieldAddInSupportServices.dll';
    
addin_class = NET.addAssembly(addin_path);
automation_class = NET.addAssembly(automation_path);
support_class = NET.addAssembly(support_path);
import PrincetonInstruments.LightField.AddIns.*;

lfi = lfm(true);

lfi.load_experiment('Raman10m');
disp('Ready for experiments (pending camera cooldown)')