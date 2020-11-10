clear;clc;close all

%% NR PUSCH Throughput
% This example shows how to measure the physical uplink shared channel
% (PUSCH) throughput of a 5G New Radio (NR) link, as defined by the 3GPP NR
% standard. The example implements PUSCH and uplink transport channel
% (UL-SCH). The transmitter model includes PUSCH demodulation reference
% symbols (DM-RS). The example supports both clustered delay line (CDL) and
% tapped delay line (TDL) propagation channels. You can perform perfect or
% practical synchronization and channel estimation. To reduce the total
% simulation time, you can execute the SNR points in the SNR loop in
% parallel by using the Parallel Computing Toolbox(TM).

% Copyright 2018-2019 The MathWorks, Inc.
 
%% Introduction
% This example measures the PUSCH throughput of a 5G link, as defined by
% the 3GPP NR standard [ <#15 1> ], [ <#15 2> ], [ <#15 3> ], [ <#15 4> ].
%
% The following 5G NR features are modeled:
%
% * UL-SCH transport channel coding
% * PUSCH and PUSCH DM-RS generation
% * Variable subcarrier spacing and frame numerologies (2^n * 15kHz)
% * Normal and extended cyclic prefix
% * TDL and CDL propagation channel models
%
% Other features of the simulation are:
%
% * Codebook and non-codebook based PUSCH transmission schemes
% * Optional PUSCH transform precoding
% * Slot wise and non slot wise PUSCH and DM-RS mapping
% * Perfect or practical synchronization and channel estimation
% * HARQ operation with 16 processes
%
% The figure below shows the processing chain implemented. For clarity, the 
% DM-RS generation has been omitted.
%
% <<../PUSCHLinkExampleProcessingChain.png>>
%
% Note that this example does not include closed-loop adaptation of the
% MIMO precoding according to channel conditions. The PUSCH MIMO precoding
% used in the example is as follows:
%
% * For codebook based transmission, the MIMO precoding matrix used inside 
% the PUSCH modulation can be selected using the TPMI parameter. 
% * The implementation-specific MIMO precoding matrix (for non-codebook
% based transmission, or MIMO precoding between transmission antenna ports
% and antennas for codebook based transmission) is an identity matrix.
%
% To reduce the total simulation time, you can use the Parallel Computing
% Toolbox to execute the SNR points of the SNR loop in parallel.

%% Simulation Length and SNR Points
% Set the length of the simulation in terms of the number of 10ms frames. A
% large number of NFrames should be used to produce meaningful throughput
% results. Set the SNR points to simulate. The SNR is defined per RE and
% applies to each receive antenna.

simParameters = [];             % Clear simParameters variable

%%
% The variable |displaySimulationInformation| controls the display of
% simulation information such as the HARQ process ID used for each
% subframe. In case of CRC error, the value of the index to the RV sequence
% is also displayed.

displaySimulationInformation = true;

%% Channel Estimator Configuration
% The logical variable |perfectChannelEstimator| controls channel
% estimation and synchronization behavior. When set to |true|, perfect
% channel estimation and synchronization is used. Otherwise, practical
% channel estimation and synchronization is used, based on the values of
% the received PUSCH DM-RS.

perfectChannelEstimator = false;

%% UE and PUSCH Configuration
% Set the key parameters of the simulation. These include:
% 
% * The bandwidth in resource blocks (12 subcarriers per resource block)
% * Subcarrier spacing: 15, 30, 60, 120, 240 (kHz)
% * Cyclic prefix length: normal or extended
% * Cell ID
% * Number of transmit and receive antennas
% 
% A substructure containing the UL-SCH and PUSCH parameters is also
% specified. This includes:
% 
% * Target code rate
% * Allocated resource blocks (PRBSet)
% * Modulation scheme: 'pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'
% * Number of layers
% * Transform precoding (enable/disable)
% * PUSCH transmission scheme and MIMO precoding matrix indication (TPMI)
% * Number of antenna ports
% * PUSCH mapping type
% * DM-RS configuration parameters
% 
% Other simulation wide parameters are:
% 
% * Propagation channel model: 'TDL' or 'CDL'
%
% Note that if transform precoding is enabled, the number of layers should
% be set to 1.

% Bandwidth, numerology (SCS and CP type) and other general parameters
simParameters.NRB = 273;                % Bandwidth in number of resource blocks (52RBs at 15kHz SCS for 10MHz BW)
simParameters.SubcarrierSpacing = 30;  % 15, 30, 60, 120, 240 (kHz)
simParameters.CyclicPrefix = 'Normal'; % 'Normal' or 'Extended'
simParameters.NCellID = 0;             % Cell identity
simParameters.NTxAnts = 1;             % Number of transmit antennas
simParameters.NRxAnts = 1;             % Number of receive antennas

% UL-SCH/PUSCH parameters
simParameters.PUSCH.PRBSet = (0:simParameters.NRB-1); % PUSCH PRB allocation
simParameters.PUSCH.SymbolSet = 0:13;            % PUSCH symbol allocation in each slot
simParameters.PUSCH.NohPRB = 0;                  % Additional RE overhead per PRB
simParameters.PUSCH.NLayers = 1;                 % Number of PUSCH layers
simParameters.PUSCH.RNTI = 1;                    % Radio Network Temporary Identifier
simParameters.PUSCH.TransformPrecoding = false;  % Enable/disable transform precoding
simParameters.PUSCH.TxScheme = 'nonCodebook';    % Transmission scheme ('nonCodebook','codebook')
simParameters.PUSCH.NAntennaPorts = 1;           % Number of antenna ports for codebook based precoding
simParameters.PUSCH.TPMI = 0;                    % Precoding matrix indicator for codebook based precoding
% PUSCH DM-RS configuration
simParameters.PUSCH.PUSCHMappingType = 'A';      % PUSCH mapping type ('A'(slot-wise),'B'(non slot-wise))
simParameters.PUSCH.DMRSTypeAPosition = 2;       % Mapping type A only. First DM-RS symbol position (2,3)
simParameters.PUSCH.DMRSLength = 1;              % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
simParameters.PUSCH.DMRSAdditionalPosition = 1;  % Additional DM-RS symbol positions (max range 0...3)
simParameters.PUSCH.DMRSConfigurationType = 1;   % DM-RS configuration type (1,2)
simParameters.PUSCH.NumCDMGroupsWithoutData = 2; % CDM groups without data
simParameters.PUSCH.NIDNSCID = 0;                % Scrambling identity (0...65535)
simParameters.PUSCH.NSCID = 0;                   % Scrambling initialization (0,1)
simParameters.PUSCH.NRSID = 0;                   % Scrambling ID for low-PAPR sequences (0...1007)
simParameters.PUSCH.GroupHopping = 'Disable';    % Hopping type ('Enable','Disable')

% Define the propagation channel type
simParameters.ChannelType = 'TDL'; % 'CDL' or 'TDL'

%% outter loop for simulation iterations with different Modulation schemes and Code Rates 
% simulation configurations
MCS_table = 2;
MCS_idx = 27;
% inner loop for SNR iterations
simParameters.SNRIn = 26:0.2:32;%[-5 0 5]; % SNR range (dB)
HARQ_status_sim = [0];
simParameters.NFrames = 10;      % Number of 10ms frames
% Array to store the maximum throughput for all SNR points
maxThroughput = zeros(length(HARQ_status_sim),length(simParameters.SNRIn)); 
% Array to store the simulation throughput for all SNR points
simThroughput = zeros(length(HARQ_status_sim),length(simParameters.SNRIn));
[ModulationType, CodingRate]=MCSidx2MCS(MCS_table,MCS_idx);
simParameters.PUSCH.TargetCodeRate = CodingRate;  % Code rate used to calculate transport block sizes
simParameters.PUSCH.Modulation = ModulationType;       % 'QPSK', '16QAM', '64QAM', '256QAM'
%%
for harq_run = 1:length(HARQ_status_sim)
        simParameters.PUSCH.EnableHARQ = HARQ_status_sim(harq_run);
        fprintf('\nSimulating MCS Table %d, Index %d with %s Modulation Type and %d Code Rate, HARQ = %d\n',...
                MCS_table, MCS_idx, ModulationType,CodingRate, simParameters.PUSCH.EnableHARQ); 
        %%
        % Create UE configuration structure |ue| and PUSCH configuration structure 
        % |pusch|.

        ue = simParameters;
        pusch = simParameters.PUSCH;

        %%
        % For key simulation parameters, define local variables for convenience.

        snrIn = simParameters.SNRIn;
        nTxAnts = simParameters.NTxAnts;
        nRxAnts = simParameters.NRxAnts;
        channelType = simParameters.ChannelType;

        %% Propagation Channel Model Configuration
        % Create the channel model object. Both CDL and TDL channel models are
        % supported [ <#15 5> ].

        if strcmpi(channelType,'CDL')

            channel = nrCDLChannel;
            channel.DelayProfile = 'CDL-A';
            [txsize,rxsize] = hArrayGeometry(nTxAnts,nRxAnts,'uplink');
            channel.TransmitAntennaArray.Size = txsize;
            channel.ReceiveAntennaArray.Size = rxsize;

        else

            channel = nrTDLChannel;
            channel.DelayProfile = 'TDL-D'; % TDL-A/B/C for NLOS, TDL-D/E for LOS
            channel.NumTransmitAntennas = nTxAnts;
            channel.NumReceiveAntennas = nRxAnts;

        end

        channel.DelaySpread = 30e-9; % in seconds
        channel.MaximumDopplerShift = 10; % in Hz

        %%
        % The sampling rate for the channel model is set using the value returned 
        % from hOFDMInfo.

        waveformInfo = hOFDMInfo(ue);
        channel.SampleRate = waveformInfo.SamplingRate;

        %%
        % Get the maximum number of delayed samples by a channel multipath
        % component. This is calculated from the channel path with the largest
        % delay and the implementation delay of the channel filter. This is
        % required later to flush the channel filter to obtain the received signal.

        chInfo = info(channel);
        maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate));
        maxChDelay = maxChDelay + chInfo.ChannelFilterDelay;

        %% Processing Loop
        % To determine the throughput at each SNR point, the PUSCH data is analyzed 
        % per transmission instance using the following steps:
        % 
        % * _Update current HARQ process._ Check the CRC of the previous
        % transmission for the given HARQ process. Determine whether a
        % retransmission is required. If that is not the case generate new data.
        % * _Generate resource grid._ Channel coding is performed by
        % <docid:5g_ref#mw_sysobj_nrULSCH nrULSCH>. It operates on the input
        % transport block provided. Internally, it keeps a copy of the transport
        % block in case a retransmission is required. The coded bits are modulated
        % by <docid:5g_ref#mw_function_nrPUSCH nrPUSCH>. Implementation-specific 
        % MIMO precoding is applied to the resulting signal. Note that if 
        % |TxScheme='codebook'|, codebook based MIMO precoding will already have
        % been applied inside |nrPUSCH| and the implementation-specific MIMO 
        % precoding is an additional stage of MIMO precoding. 
        % * _Generate waveform._ The generated grid is then OFDM modulated.
        % * _Model noisy channel._ The waveform is passed through a CDL or TDL
        % fading channel. AWGN is added. The SNR for each layer is defined per RE
        % and per receive antenna.
        % * _Perform synchronization and OFDM demodulation._ For perfect
        % synchronization, the channel impulse response is reconstructed and used
        % to synchronize the received waveform. For practical synchronization, the
        % received waveform is correlated with the PUSCH DM-RS. The synchronized
        % signal is then OFDM demodulated.
        % * _Perform channel estimation._ If perfect channel estimation is used,
        % the channel impulse response is reconstructed and OFDM demodulated to
        % provide a channel estimate. For practical channel estimation, the
        % PUSCH DM-RS is used.
        % * _Extract PUSCH and perform equalization._ The resource elements
        % corresponding to the PUSCH allocation are extracted from the received
        % OFDM resource grid and the channel estimate using
        % <docid:5g_ref#mw_function_nrExtractResources nrExtractResources>. The
        % received PUSCH resource elements are then MMSE equalized using
        % <docid:5g_ref#mw_function_nrEqualizeMMSE nrEqualizeMMSE>.
        % * _Decode the PUSCH._ The equalized PUSCH symbols, along with a noise
        % estimate, are demodulated and descrambled by
        % <docid:5g_ref#mw_function_nrPUSCHDecode nrPUSCHDecode> to obtain an
        % estimate of the received codewords.
        % * _Decode the Uplink Shared Channel (UL-SCH) and store the block CRC
        % error for a HARQ process._ The vector of decoded soft bits is passed to
        % <docid:5g_ref#mw_sysobj_nrULSCHDecoder nrULSCHDecoder> which decodes
        % the codeword and returns the block CRC error used to determine the
        % throughput of the system.


        % Set up Redundancy Version (RV) sequence, number of HARQ processes and
        % the sequence in which the HARQ processes are used
        if pusch.EnableHARQ
            % From PUSCH demodulation requirements in RAN WG4 meeting #88bis
            % (R4-1814062)
            rvSeq = [0 1 2 3];%[0 2 3 1];
        else
            % HARQ disabled - single transmission with RV=0, no retransmissions
            rvSeq = 0;
        end

        % Create UL-SCH encoder System object
        encodeULSCH = nrULSCH;
        encodeULSCH.MultipleHARQProcesses = true;
        encodeULSCH.TargetCodeRate = pusch.TargetCodeRate;

        % Create UL-SCH decoder System object
        % Use layered belief propagation for LDPC decoding, with half the number of
        % iterations as compared to the default for belief propagation decoding
        decodeULSCH = nrULSCHDecoder;
        decodeULSCH.MultipleHARQProcesses = true;
        decodeULSCH.TargetCodeRate = pusch.TargetCodeRate;
        decodeULSCH.LDPCDecodingAlgorithm = 'Layered belief propagation';
        decodeULSCH.MaximumLDPCIterationCount = 6; 

        % The temporary variables 'ue_init', 'pusch_init' and 'decodeULSCH_init'
        % are used to create the temporary variables 'ue', 'pusch' and
        % 'decodeULSCH' within the SNR loop to create independent instances in case
        % of parallel simulation
        ue_init = ue;
        pusch_init = pusch;
        decodeULSCH_init = clone(decodeULSCH);

        for snrIdx = 1:numel(snrIn) % comment out for parallel computing
        % parfor snrIdx = 1:numel(snrIn) % uncomment for parallel computing
        % To reduce the total simulation time, you can execute this loop in
        % parallel by using the Parallel Computing Toolbox. Comment out the 'for'
        % statement and uncomment the 'parfor' statement. If the Parallel Computing
        % Toolbox is not installed, 'parfor' defaults to normal 'for' statement.
        % Because parfor-loop iterations are executed in parallel in a
        % nondeterministic order, the simulation information displayed for each SNR
        % point can be intertwined. To switch off simulation information display,
        % set the 'displaySimulationInformation' variable above to false

            % Reset the random number generator and channel, so that each SNR point
            % will experience the same noise and channel realizations
            rng('default');
            reset(channel);

            % Initialize variables for this SNR point, required for initialization
            % of variables when using the Parallel Computing Toolbox
            ue = ue_init;
            pusch = pusch_init;
            pathFilters = [];

            % Specify the order in which we cycle through the HARQ processes
            NHARQProcesses = 5;%16;
            harqSequence = 1:NHARQProcesses;

            % Initialize the state of all HARQ processes and reset the UL-SCH 
            % decoder
            harqProcesses = hNewHARQProcesses(NHARQProcesses,rvSeq,1);
            harqProcCntr = 0; % HARQ process counter
            decodeULSCH = clone(decodeULSCH_init);

            SNRdB = snrIn(snrIdx);
            fprintf('\nSimulating %s-based transmission scheme (%dx%d) and SCS=%dkHz with %s channel at %gdB SNR for %d 10ms frame(s)\n',pusch.TxScheme,nTxAnts,nRxAnts,ue.SubcarrierSpacing,channelType,SNRdB,ue.NFrames);

            % Total number of OFDM symbols in the simulation period
            waveformInfo = hOFDMInfo(ue);
            NSymbols = ue.NFrames * 10 * waveformInfo.SymbolsPerSubframe;

            % OFDM symbol number associated with start of each PUSCH transmission
            ue.NSymbol = 0;

            % Running counter of the number of PUSCH transmission instances
            % The simulation will use this counter as the slot number for each
            % PUSCH
            pusch.NSlot = 0;

            % Timing offset, updated in every slot for perfect synchronization and
            % when the correlation is strong for practical synchronization
            offset = 0;

            while ue.NSymbol < NSymbols

                % Calculate the transport block size for this slot
                [puschIndices,dmrsIndices,dmrsSymbols,puschIndicesInfo] = hPUSCHResources(ue,pusch);
                TBS = hPUSCHTBS(pusch,puschIndicesInfo.NREPerPRB - pusch.NohPRB);

                % Get HARQ process index for the current PUSCH from the HARQ index
                % table
                harqProcIdx = harqSequence(mod(harqProcCntr,length(harqSequence))+1);

                % Update current HARQ process information (this updates the RV
                % depending on CRC pass or fail in the previous transmission for
                % this HARQ process)
                harqProcesses(harqProcIdx) = hUpdateHARQProcess(harqProcesses(harqProcIdx),1);

                % HARQ: check CRC from previous transmission, i.e. is a
                % retransmission required?
                NDI = false;
                if harqProcesses(harqProcIdx).blkerr % errored
                    if (harqProcesses(harqProcIdx).RVIdx==1) % end of rvSeq
                        resetSoftBuffer(decodeULSCH,harqProcIdx-1);
                        NDI = true;
                    end
                else % no error
                    NDI = true;
                end
                if NDI 
                    trBlk = randi([0 1],TBS,1);
                    setTransportBlock(encodeULSCH,trBlk,harqProcIdx-1);
                end

                % UL-SCH encoding
                codedTrBlock = encodeULSCH(pusch.Modulation,pusch.NLayers,puschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);

                % PUSCH modulation, including codebook based MIMO precoding if 
                % TxScheme = 'codebook'
                MRB = numel(pusch.PRBSet);
                puschSymbols = nrPUSCH(codedTrBlock,pusch.Modulation,pusch.NLayers,ue.NCellID,pusch.RNTI,pusch.TransformPrecoding,MRB,pusch.TxScheme,pusch.NAntennaPorts,pusch.TPMI);

                % Create resource grid associated with PUSCH transmission period
                puschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);

                % Implementation-specific PUSCH MIMO precoding and mapping. This 
                % MIMO precoding step is in addition to any codebook based 
                % MIMO precoding done during PUSCH modulation above
                if (strcmpi(pusch.TxScheme,'codebook'))
                    % codebook based MIMO precoding, F precodes between PUSCH
                    % transmit antenna ports and transmit antennas
                    F = eye(pusch.NAntennaPorts,nTxAnts);
                else
                    % non-codebook based MIMO precoding, F precodes between PUSCH 
                    % layers and transmit antennas
                    F = eye(pusch.NLayers,nTxAnts);
                end
                [~,puschAntIndices] = nrExtractResources(puschIndices,puschGrid);
                puschGrid(puschAntIndices) = puschSymbols * F;

                % Implementation-specific PUSCH DM-RS MIMO precoding and mapping.
                % The DM-RS creation in hPUSCHResources above includes codebook
                % based MIMO precoding if applicable
                for p = 1:size(dmrsSymbols,2)
                    [~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),puschGrid);
                    puschGrid(dmrsAntIndices) = puschGrid(dmrsAntIndices) + dmrsSymbols(:,p) * F(p,:);
                end

                % OFDM modulation
                txWaveform = hOFDMModulate(ue,puschGrid);
                rxWaveform = txWaveform;
                % Pass data through channel model. Append zeros at the end of the
                % transmitted waveform to flush channel content. These zeros take
                % into account any delay introduced in the channel. This is a mix
                % of multipath delay and implementation delay. This value may 
                % change depending on the sampling rate, delay profile and delay
                % spread
%                 txWaveform = [txWaveform; zeros(maxChDelay,size(txWaveform,2))]; %#ok<AGROW>
%                 [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
                

                % Add AWGN to the received time domain waveform 
                % Normalize noise power by the IFFT size used in OFDM modulation,
                % as the OFDM modulator applies this normalization to the
                % transmitted waveform. Also normalize by the number of receive
                % antennas, as the default behaviour of the channel model is to
                % apply this normalization to the received waveform
                SNR = 10^(SNRdB/20);
                N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
                noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
                rxWaveform = rxWaveform + noise;

                if (perfectChannelEstimator)
                    % Perfect synchronization. Use information provided by the
                    % channel to find the strongest multipath component
                    pathFilters = getPathFilters(channel);
                    [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
                else
                    % Practical synchronization. Correlate the received waveform 
                    % with the PUSCH DM-RS to give timing offset estimate 't' and 
                    % correlation magnitude 'mag'. The function
                    % hSkipWeakTimingOffset is used to update the receiver timing
                    % offset. If the correlation peak in 'mag' is weak, the current
                    % timing estimate 't' is ignored and the previous estimate
                    % 'offset' is used
                    [t,mag] = nrTimingEstimate(rxWaveform,ue.NRB,ue.SubcarrierSpacing,pusch.NSlot,dmrsIndices,dmrsSymbols,'CyclicPrefix',ue.CyclicPrefix); %#ok<UNRCH>
                    offset = hSkipWeakTimingOffset(offset,t,mag);
                end
                rxWaveform = rxWaveform(1+offset:end,:);

                % Perform OFDM demodulation on the received data to recreate the
                % resource grid, including padding in the event that practical
                % synchronization results in an incomplete slot being demodulated
                rxGrid = hOFDMDemodulate(ue,rxWaveform);
                [K,L,R] = size(rxGrid);
                if (L < waveformInfo.SymbolsPerSlot)
                    rxGrid = cat(2,rxGrid,zeros(K,waveformInfo.SymbolsPerSlot-L,R));
                end

                if (perfectChannelEstimator)
                    % Perfect channel estimation, use the value of the path gains
                    % provided by the channel
                    estChannelGrid = nrPerfectChannelEstimate(pathGains,pathFilters,ue.NRB,ue.SubcarrierSpacing,pusch.NSlot,offset,sampleTimes,ue.CyclicPrefix);

                    % Get perfect noise estimate (from the noise realization)
                    noiseGrid = hOFDMDemodulate(ue,noise(1+offset:end,:));
                    noiseEst = var(noiseGrid(:));

                    % Apply MIMO deprecoding to estChannelGrid to give an estimate
                    % per transmission layer
                    K = size(estChannelGrid,1);
                    estChannelGrid = reshape(estChannelGrid,K*waveformInfo.SymbolsPerSlot*nRxAnts,nTxAnts);
                    estChannelGrid = estChannelGrid * F.';
                    if (strcmpi(pusch.TxScheme,'codebook'))
                        W = nrPUSCHCodebook(pusch.NLayers,pusch.NAntennaPorts,pusch.TPMI,pusch.TransformPrecoding);
                        estChannelGrid = estChannelGrid * W.';
                    end
                    estChannelGrid = reshape(estChannelGrid,K,waveformInfo.SymbolsPerSlot,nRxAnts,[]);
                else
                    % Practical channel estimation between the received grid and
                    % each transmission layer, using the PUSCH DM-RS for each layer
                    [~,dmrsLayerIndices,dmrsLayerSymbols] = hPUSCHResources(ue,setfield(pusch,'TxScheme','nonCodebook')); %#ok<UNRCH>
                    [estChannelGrid,noiseEst] = nrChannelEstimate(rxGrid,dmrsLayerIndices,dmrsLayerSymbols,'CyclicPrefix',ue.CyclicPrefix,'CDMLengths',puschIndicesInfo.CDMLengths);
                end

                % Get PUSCH resource elements from the received grid
                [puschRx,puschHest] = nrExtractResources(puschIndices,rxGrid,estChannelGrid);

                % Equalization
                [puschEq,csi] = nrEqualizeMMSE(puschRx,puschHest,noiseEst);

                % Decode PUSCH physical channel
                [ulschLLRs,rxSymbols] = nrPUSCHDecode(puschEq,pusch.Modulation,ue.NCellID,pusch.RNTI,noiseEst,pusch.TransformPrecoding,MRB);

                % Apply channel state information (CSI) produced by the equalizer,
                % including the effect of transform precoding if enabled
                if (pusch.TransformPrecoding)
                    MSC = MRB * 12;
                    csi = nrTransformDeprecode(csi,MRB) / sqrt(MSC);
                    csi = repmat(csi((1:MSC:end).'),1,MSC).';
                    csi = reshape(csi,size(rxSymbols));
                end
                csi = nrLayerDemap(csi);
                Qm = length(ulschLLRs) / length(rxSymbols);
                csi = reshape(repmat(csi{1}.',Qm,1),[],1);
                ulschLLRs = ulschLLRs .* csi;

                % Decode the UL-SCH transport channel
                decodeULSCH.TransportBlockLength = TBS;
                [decbits,harqProcesses(harqProcIdx).blkerr] = decodeULSCH(ulschLLRs,pusch.Modulation,pusch.NLayers,harqProcesses(harqProcIdx).RV,harqProcIdx-1);

                % Store values to calculate throughput
                simThroughput(harq_run,snrIdx) = simThroughput(harq_run,snrIdx) + (~harqProcesses(harqProcIdx).blkerr * TBS);
                maxThroughput(harq_run,snrIdx) = maxThroughput(harq_run,snrIdx) + TBS;

                % Display transport block error information
                if (displaySimulationInformation)
                    fprintf('\n(%3.2f%%) HARQ Proc %d: ',100*(ue.NSymbol+size(puschGrid,2))/NSymbols,harqProcIdx);
                    estrings = {'passed','failed'};
                    rvi = harqProcesses(harqProcIdx).RVIdx;
                    if rvi == 1
                        ts = sprintf('Initial transmission (RV=%d)',rvSeq(rvi));
                    else
                        ts = sprintf('Retransmission #%d (RV=%d)',rvi-1,rvSeq(rvi));
                    end
                    fprintf('%s %s. ',ts,estrings{1+harqProcesses(harqProcIdx).blkerr});
                end

                % Update starting symbol number of next PUSCH transmission
                ue.NSymbol = ue.NSymbol + size(puschGrid,2);

                % Update count of overall number of PUSCH transmissions
                pusch.NSlot = pusch.NSlot + 1;

                % Update HARQ process counter
                harqProcCntr = harqProcCntr + 1;

            end

            % Display the results dynamically in the command window
            if (displaySimulationInformation)
                fprintf('\n');
            end
            fprintf([['\nThroughput(Mbps) for ' num2str(ue.NFrames) ' frame(s) '],'= %.4f\n'], 1e-6*simThroughput(harq_run,snrIdx)/(ue.NFrames*10e-3));
            fprintf(['Throughput(%%) for ' num2str(ue.NFrames) ' frame(s) = %.4f\n'],simThroughput(harq_run,snrIdx)*100/maxThroughput(harq_run,snrIdx));

        end
end
%% Results
% Display the measured throughput. This is calculated as the percentage of
% the maximum possible throughput of the link given the available resources
% for data transmission.
cd Sim_results\
save simThroughput_MCS27_noHARQ.mat simThroughput
save maxThroughput_MCS27_noHARQ.mat maxThroughput
save simParameters_MCS27_noHARQ.mat simParameters
cd ..

% cd Sim_results\
% save simThroughput_withHARQ_1T1R_table2.mat simThroughput
% save maxThroughput_withHARQ_1T1R_table2.mat maxThroughput
% save simParameters_withHARQ_1T1R_table2.mat simParameters
% cd ..

figure;
hold on
for harq_run = 1:length(HARQ_status_sim)
    plot(snrIn,(100 - simThroughput(harq_run,:)*100./maxThroughput(harq_run,:)),'o-.','LineWidth', 2)
end
line([min(snrIn), max(snrIn)],[10, 10],'Color','red','LineStyle','-','LineWidth',1)
line([min(snrIn), max(snrIn)],[1, 1],'Color','red','LineStyle','-','LineWidth',1)
hold off
xlabel('SNR (dB)'); ylabel('BLER (%)'); grid on;
% title(sprintf('UL BLER vs SNR, MCS table-2, index 20, 1T1R'));
if (pusch_init.TransformPrecoding)
    ofdmType = 'DFT-s-OFDM';
else
    ofdmType = 'CP-OFDM';
end
title(sprintf('%s / NRB=%d / SCS=%dkHz / %s %d/1024 / %dx%d', ...
    ofdmType,ue_init.NRB,ue_init.SubcarrierSpacing, ...
    pusch_init.Modulation, ...
    round(pusch_init.TargetCodeRate*1024),nTxAnts,nRxAnts));

simResults.simParameters = simParameters;
simResults.simThroughput = simThroughput;
simResults.maxThroughput = maxThroughput;

%%
% The figure below shows throughput results obtained simulating 10000
% subframes (|NFrames = 1000|, |SNRIn = -16:2:6|).
%
% <<../longRunPUSCHThroughput.png>>
%

%% Appendix
% This example uses the following helper functions:
%
% * <matlab:edit('hArrayGeometry.m') hArrayGeometry.m>
% * <matlab:edit('hNewHARQProcesses.m') hNewHARQProcesses.m>
% * <matlab:edit('hOFDMDemodulate.m') hOFDMDemodulate.m>
% * <matlab:edit('hOFDMInfo.m') hOFDMInfo.m>
% * <matlab:edit('hOFDMModulate.m') hOFDMModulate.m>
% * <matlab:edit('hPUSCHTBS.m') hPUSCHTBS.m>
% * <matlab:edit('hPUSCHResources.m') hPUSCHResources.m>
% * <matlab:edit('hSkipWeakTimingOffset.m') hSkipWeakTimingOffset.m>
% * <matlab:edit('hUpdateHARQProcess.m') hUpdateHARQProcess.m>

%% Selected Bibliography
% # 3GPP TS 38.211. "NR; Physical channels and modulation (Release 15)."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.212. "NR; Multiplexing and channel coding (Release 15)." 3rd
% Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.213. "NR; Physical layer procedures for control (Release
% 15)." 3rd Generation Partnership Project; Technical Specification Group
% Radio Access Network.
% # 3GPP TS 38.214. "NR; Physical layer procedures for data (Release 15)."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TR 38.901. "Study on channel model for frequencies from 0.5 to 100
% GHz (Release 15)." 3rd Generation Partnership Project; Technical
% Specification Group Radio Access Network.
