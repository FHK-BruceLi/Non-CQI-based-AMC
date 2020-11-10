function [ModulationType, CodingRate]=MCSidx2MCS(MCStable,MCS_idx)
%% load MCSindex-to-Modulation&coding rate tables, TS-38.214 Table 5.1.3.1-1/2/3, without UL transform precoding
% [MCS index, Modulation type, Coding rate]
switch MCStable
    case 1
        load MCS_table_1.mat
        MCS_table = MCS_table_1;
    case 2
        load MCS_table_2.mat
        MCS_table = MCS_table_2;
%     case 3
%         load MCS_table_3.mat
%         MCS_table = MCS_table_3;
end
%%
idx = MCS_idx + 1;
ModulationType_idx = MCS_table(idx, 2);
switch ModulationType_idx
    case 2
        ModulationType = 'QPSK';
    case 4
        ModulationType = '16QAM';
    case 6
        ModulationType = '64QAM';
    case 8
        ModulationType = '256QAM';        
end
CodingRate = MCS_table(idx, 3);
end
