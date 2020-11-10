simParameters = [];             % Clear simParameters variable
perfectChannelEstimator = false;

simParameters.NRB = 273;                  % Bandwidth in number of resource blocks (51RBs at 30kHz SCS for 20MHz BW)
simParameters.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
simParameters.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended'

simParameters.NCellID = 1;               % Cell identity

% DL-SCH/PDSCH parameters
simParameters.PDSCH.PRBSet = 0:simParameters.NRB-1; % PDSCH PRB allocation
simParameters.PDSCH.SymbolSet = 0:13;           % PDSCH symbol allocation in each slot
simParameters.PDSCH.EnableHARQ = false;%true;          % Enable/disable HARQ, if disabled, single transmission with RV=0, i.e. no retransmissions

simParameters.PDSCH.NLayers = 1;                % Number of PDSCH layers
simParameters.NTxAnts = 1;                      % Number of PDSCH transmission antennas

simParameters.NumCW = 1;                        % Number of codewords
simParameters.NRxAnts = 1;                      % Number of UE receive antennas

% DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
simParameters.PDSCH.PortSet = 0:simParameters.PDSCH.NLayers-1; % DM-RS ports to use for the layers
simParameters.PDSCH.PDSCHMappingType = 'A';     % PDSCH mapping type ('A'(slot-wise),'B'(non slot-wise))
simParameters.PDSCH.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
simParameters.PDSCH.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
simParameters.PDSCH.DMRSAdditionalPosition = 0; % Additional DM-RS symbol positions (max range 0...3)
simParameters.PDSCH.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
simParameters.PDSCH.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
simParameters.PDSCH.NIDNSCID = 1;               % Scrambling identity (0...65535)
simParameters.PDSCH.NSCID = 0;                  % Scrambling initialization (0,1)
% Reserved PRB patterns (for CORESETs, forward compatibility etc)
simParameters.PDSCH.Reserved.Symbols = [];      % Reserved PDSCH symbols
simParameters.PDSCH.Reserved.PRB = [];          % Reserved PDSCH PRBs
simParameters.PDSCH.Reserved.Period = [];       % Periodicity of reserved resources
% PDSCH resource block mapping (TS 38.211 Section 7.3.1.6)
simParameters.PDSCH.VRBToPRBInterleaving = 0;   % Disable interleaved resource mapping

% Define the propagation channel type
simParameters.ChannelType = 'CDL'; % 'CDL' or 'TDL'

% SS burst configuration
simParameters.SSBurst.BlockPattern = 'Case B';    % 30kHz subcarrier spacing
simParameters.SSBurst.SSBTransmitted = [0 0 0 1]; % Bitmap indicating blocks transmitted in the burst
simParameters.SSBurst.SSBPeriodicity = 20;        % SS burst set periodicity in ms (5, 10, 20, 40, 80, 160)

% validateNLayers(simParameters);
