function [ModulationType, MCS_index, CodingRate] = Adaptive_MCS_update(sinr, bler, CQI_table)

%% for lack of CQI, first get the baseline MCS and coding rate according to SINR
[ModulationType_SINR, MCS_index_SINR, CodingRate_SINR]=SINR2indexMCS(sinr);

%% adpatively raise or lower MCS and coding rate according to BLER 
%  (threshold for table 1&2 is 0.1, 0.00001 for table 3, TS-38.214 Table 5.2.2.1-2/3/4)
switch CQI_table
    case 1
        TH = 0.1;
    case 2
        TH = 0.1;
    case 3
        TH = 0.00001;
end

end