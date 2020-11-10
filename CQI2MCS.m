function [ModulationType, CodingRate, Spectral_efficiency]=CQI2MCS(CQI_table,CQI_idx)
%% load CQI-to-Modulation&coding rate tables, TS-38.214 Table 5.2.2.1-2/3/4
% [CQI index, Modulation type, Coding rate, Spectral efficiency]
switch CQI_table
    case 1
        load CQI_table_1.mat
        CQI_table = CQI_table_1;
    case 2
        load CQI_table_2.mat
        CQI_table = CQI_table_2;
    case 3
        load CQI_table_3.mat
        CQI_table = CQI_table_3;
end
%%
idx = CQI_idx + 1;
ModulationType_idx = CQI_table(idx, 2);
switch ModulationType_idx
    case 1
        ModulationType = 'QPSK';
    case 2
        ModulationType = '16QAM';
    case 3
        ModulationType = '64QAM';
    case 4
        ModulationType = '256QAM';        
end
CodingRate = CQI_table(idx, 3);
Spectral_efficiency = CQI_table(idx, 4);
end
