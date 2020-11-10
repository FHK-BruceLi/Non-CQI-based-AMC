function [ModulationType, MCS_index, CodingRate]=SINR2MCS(sinr)
%#codegen
% Table of SINR threshold values, 27 boundary points for an unbounded quantizer
thresh=[-10, -8, -6.7,-4.7,-2.3,0.2,2.4,4.3,5.9,8.1,10.3,11.7,14.1,16.3,18.7,21,22.7, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43];
% Table of coding rate (28 value, TS 38.214 Table 5.1.3.1-2, MCS index table 2 for PDSCH)
Map2CodingRate=[0.1171875, 0.1884765625, 0.30078125, 0.4384765625, 0.587890625,...   % QPSK 0-4
    0.369140625, 0.423828125, 0.478515625, 0.5400390625, 0.6015625, 0.642578125,...  % 16QAM 5-10
    0.455078125, 0.5048828125, 0.5537109375, 0.6015625, 0.650390625, 0.7021484375, 0.75390625, 0.802734375, 0.8525390625,... % 64QAM 11-19
    0.66650390625, 0.6943359375, 0.736328125, 0.7783203125, 0.8212890625, 0.8642578125, 0.89501953125, 0.92578125]; % 256QAM 20-27
% Table of modulation type (1=QPSK, 2=16QAM, 3=64QAM, 4=256QAM)
Map2Modulator=[1*ones(5,1);2*ones(6,1);3*ones(9,1);4*ones(8,1)];
persistent hQ
if isempty(hQ)
hQ=dsp.ScalarQuantizerEncoder('Partitioning', 'Unbounded','BoundaryPoints', thresh,'OutputIndexDataType','uint8');
end
MCS_index=step(hQ, sinr);
index1=MCS_index+1; % 1-based indexing
% Map MCS index to modulation type
ModulationType_idx = Map2Modulator (index1);
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
% Map CQI index to coding rate
CodingRate = Map2CodingRate (index1);
end