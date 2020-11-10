clear;clc;close all
clear DL_AMC_AN_alg1
clear AN_Counter_alg1
clear AN_Counter_alg2

Sim_Params_Init; % load initial simulation parameters
Sim_iter = 2e3;
simParameters.nTTI = 1;      % Number of slots every simulation iteration
SNR_mean = 14;
[ModulationType, MCS_idx, CodingRate]=SINR2MCS(SNR_mean);
%% get Modulation type and Coding rate 
MCS_table = 2;
% MCS_idx = 27;
% [ModulationType, CodingRate] = MCSidx2MCS(MCS_table,MCS_idx);
simParameters.PDSCH.TargetCodeRate = CodingRate;
simParameters.PDSCH.Modulation = ModulationType; 
% %% build OTA channel
% nTxAnts = simParameters.NTxAnts;
% nRxAnts = simParameters.NRxAnts;
% channelType = simParameters.ChannelType;
% if strcmpi(channelType,'CDL')
%     channel = nrCDLChannel; % CDL channel object
%     % Use CDL-C model (Urban macrocell model)
%     channel.DelayProfile = 'CDL-C';
%     channel.DelaySpread = 300e-9;
% 
%     % Turn the overall number of antennas into a specific antenna panel
%     % array geometry
%     [channel.TransmitAntennaArray.Size, channel.ReceiveAntennaArray.Size] = ...
%         hArrayGeometry(nTxAnts,nRxAnts);
% elseif strcmpi(channelType,'TDL')
%     channel = nrTDLChannel; % TDL channel object
%     % Set the channel geometry
%     channel.DelayProfile = 'TDL-C';
%     channel.DelaySpread = 300e-9;
%     channel.NumTransmitAntennas = nTxAnts;
%     channel.NumReceiveAntennas = nRxAnts;
% else
%     error('ChannelType parameter field must be either CDL or TDL.');
% end
%% 
gnb = simParameters;
pdsch = simParameters.PDSCH;
pdsch.RNTI = 1;
Xoh_PDSCH = 0;     % The Xoh-PDSCH overhead value is taken to be 0 here
waveformInfo = hOFDMInfo(gnb);
% channel.SampleRate = waveformInfo.SamplingRate;
% %% 
% % Get the maximum number of delayed samples by a channel multipath
% % component. This is calculated from the channel path with the largest
% % delay and the implementation delay of the channel filter. This is
% % required later to flush the channel filter to obtain the received signal.
% chInfo = info(channel);
% maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
%% Reserve PDSCH Resources Corresponding to SS burst
% This section shows how to reserve resources for the transmission of the 
% SS burst.
% Create SS burst information structure
ssburst = simParameters.SSBurst;
ssburst.NCellID = gnb.NCellID;
ssburst.SampleRate = waveformInfo.SamplingRate;
ssbInfo = hSSBurstInfo(ssburst);
% Map the occupied subcarriers and transmitted symbols of the SS burst
% (defined in the SS burst numerology) to PDSCH PRBs and symbols in the
% data numerology
[mappedPRB,mappedSymbols] = mapNumerology(ssbInfo.OccupiedSubcarriers,ssbInfo.OccupiedSymbols,ssbInfo.NRB,gnb.NRB,ssbInfo.SubcarrierSpacing,gnb.SubcarrierSpacing);
% Configure the PDSCH to reserve these resources so that the PDSCH
% transmission does not overlap the SS burst
reservation.Symbols = mappedSymbols;
reservation.PRB = mappedPRB;
reservation.Period = ssburst.SSBPeriodicity * (gnb.SubcarrierSpacing/15); % Period in slots
pdsch.Reserved(end+1) = reservation;

if pdsch.EnableHARQ
    rvSeq = [0 2 3 1];
else
    rvSeq = 0; 
end
% Create DLSCH encoder system object
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = true;
encodeDLSCH.TargetCodeRate = pdsch.TargetCodeRate;
% Create DLSCH decoder system object
% Use layered belief propagation for LDPC decoding, with half the number of
% iterations as compared to the default for belief propagation decoding
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = true;
decodeDLSCH.TargetCodeRate = pdsch.TargetCodeRate;
decodeDLSCH.LDPCDecodingAlgorithm = 'Layered belief propagation';
decodeDLSCH.MaximumLDPCIterationCount = 6;
% The temporary variables 'gnb_init', 'pdsch_init', 'ssburst_init' and
% 'decodeDLSCH_init' are used to create the temporary variables 'gnb',
% 'pdsch', 'ssburst' and 'decodeDLSCH' within the SNR loop to create
% independent instances in case of parallel simulation
% gnb_init = gnb;
% pdsch_init = pdsch;
% ssburst_init = ssburst;
% decodeDLSCH_init = clone(decodeDLSCH);
%% Processing Loop
maxThroughput = zeros(Sim_iter, 2); 
simThroughput = zeros(Sim_iter, 2);
MCSidx_record_alg1 = zeros(Sim_iter, 1);
MCSidx_record_alg2 = zeros(Sim_iter, 1);
MCS_alg1 = MCS_idx;
MCS_alg2 = MCS_idx;
for Alg_mode = 1:2
    pdsch_AlgSim(Alg_mode) = pdsch;
%     encodeDLSCH_AlgSim(Alg_mode) = encodeDLSCH;
%     decodeDLSCH_AlgSim(Alg_mode) = decodeDLSCH;
end
encodeDLSCH_alg1 = encodeDLSCH;
encodeDLSCH_alg2 = encodeDLSCH;
decodeDLSCH_alg1 = decodeDLSCH;
decodeDLSCH_alg2 = decodeDLSCH;
% SNR_variation = zeros(Sim_iter, 1);
% for iter = 1:Sim_iter
%     SNR_variation(iter) = 0.1*randi([-5, 5]); % SNR randomly and slightly varies between simulation iterations
% end
% load SNR_variation_10000iter.mat
for iter = 1:Sim_iter
    
    MCSidx_record_alg1(iter) = MCS_alg1;
    MCSidx_record_alg2(iter) = MCS_alg2;
%% build OTA channel
nTxAnts = simParameters.NTxAnts;
nRxAnts = simParameters.NRxAnts;
channelType = simParameters.ChannelType;
if strcmpi(channelType,'CDL')
    channel = nrCDLChannel; % CDL channel object
    % Use CDL-C model (Urban macrocell model)
    channel.DelayProfile = 'CDL-C';
    channel.DelaySpread = 300e-9;

    % Turn the overall number of antennas into a specific antenna panel
    % array geometry
    [channel.TransmitAntennaArray.Size, channel.ReceiveAntennaArray.Size] = ...
        hArrayGeometry(nTxAnts,nRxAnts);
elseif strcmpi(channelType,'TDL')
    channel = nrTDLChannel; % TDL channel object
    % Set the channel geometry
    channel.DelayProfile = 'TDL-C';
    channel.DelaySpread = 300e-9;
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;
else
    error('ChannelType parameter field must be either CDL or TDL.');
end
channel.SampleRate = waveformInfo.SamplingRate;
%% 
% Get the maximum number of delayed samples by a channel multipath
% component. This is calculated from the channel path with the largest
% delay and the implementation delay of the channel filter. This is
% required later to flush the channel filter to obtain the received signal.
chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
%%
%     snrIn = SNR_mean + SNR_variation(iter); 
    snrIn = SNR_mean; 
%     rng('default');
    % Initialize variables for this SNR point, required for initialization
    % of variables when using the Parallel Computing Toolbox
%     gnb = gnb_init;
%     pdsch = pdsch_init;
%     ssburst = ssburst_init;
%     decodeDLSCH = clone(decodeDLSCH_init);
    pathFilters = [];
    ssbWaveform = [];

    fprintf('\n %d out of %d TTI(slot)(s), transmission scheme 1 (%dx%d) and SCS=%dkHz with %s channel at %gdB SNR\n',...
        iter,Sim_iter,nTxAnts,nRxAnts,gnb.SubcarrierSpacing, ...
        channelType,snrIn); 

    % Initialize variables used in the simulation and analysis
    bitTput = cell(1,2);           % Number of successfully received bits per transmission
    txedTrBlkSizes = cell(1,2);    % Number of transmitted info bits per transmission
    trBlkSizes = cell(1,2);
    % Specify the order in which we cycle through the HARQ processes
    NHARQProcesses = 16;
    harqSequence = 1:NHARQProcesses;
    % Initialize the state of all HARQ processes
    harqProcesses = hNewHARQProcesses(NHARQProcesses,rvSeq,gnb.NumCW);
    harqProcCntr = 0; % HARQ process counter
    % Reset the channel so that each SNR point will experience the same
    % channel realization
%     reset(channel);
    % Total number of OFDM symbols in the simulation period
    waveformInfo = hOFDMInfo(gnb);
    NSymbols = gnb.nTTI * 0.5* waveformInfo.SymbolsPerSubframe;

    % Running counter of the number of PDSCH transmission instances
    % The simulation will use this counter as the slot number for each
    % PDSCH
    for Alg_mode = 1:2
        pdsch_AlgSim(Alg_mode).NSlot = 0;
        % OFDM symbol number associated with start of each PDSCH transmission
        gnb.NSymbol = 0;
        % Index to the start of the current set of SS burst samples to be
        % transmitted
        ssbSampleIndex = 1;
        % Obtain a precoding matrix (wtx) to be used in the transmission of the
        % first transport block
        estChannelGrid = getInitialChannelEstimate(gnb,nTxAnts,channel);    
        newWtx = getPrecodingMatrix(pdsch_AlgSim(Alg_mode).PRBSet,pdsch_AlgSim(Alg_mode).NLayers,estChannelGrid);
        % Timing offset, updated in every slot for perfect synchronization and
        % when the correlation is strong for practical synchronization
        offset = 0;
                  while gnb.NSymbol < NSymbols  % Move to next slot, gnb.NSymbol increased in steps of one slot               

                    % Generate a new SS burst when necessary
                    if (ssbSampleIndex==1)        
                        nSubframe = gnb.NSymbol / waveformInfo.SymbolsPerSubframe;
                        ssburst.NFrame = floor(nSubframe / 10);
                        ssburst.NHalfFrame = mod(nSubframe / 5,2);
                        [ssbWaveform,~,ssbInfo] = hSSBurst(ssburst);
                    end

                    % Get HARQ process index for the current PDSCH from HARQ index table
                    harqProcIdx = harqSequence(mod(harqProcCntr,length(harqSequence))+1);

                    % Update current HARQ process information (this updates the RV
                    % depending on CRC pass or fail in the previous transmission for
                    % this HARQ process)
                    harqProcesses(harqProcIdx) = hUpdateHARQProcess(harqProcesses(harqProcIdx),gnb.NumCW);

                    % Calculate the transport block sizes for the codewords in the slot
                    [pdschIndices,dmrsIndices,dmrsSymbols,pdschIndicesInfo] = hPDSCHResources(gnb,pdsch_AlgSim(Alg_mode));
                    trBlkSizes{Alg_mode} = hPDSCHTBS(pdsch_AlgSim(Alg_mode),pdschIndicesInfo.NREPerPRB-Xoh_PDSCH);

                    % HARQ: check CRC from previous transmission per codeword, i.e. is
                    % a retransmission required?
                    for cwIdx = 1:gnb.NumCW
                        NDI = false;
                        if harqProcesses(harqProcIdx).blkerr(cwIdx) % Errored
                            if (harqProcesses(harqProcIdx).RVIdx(cwIdx)==1) % end of rvSeq
                                if (Alg_mode == 1)
                                    resetSoftBuffer(decodeDLSCH_alg1,cwIdx-1,harqProcIdx-1);
                                else
                                    resetSoftBuffer(decodeDLSCH_alg2,cwIdx-1,harqProcIdx-1);
                                end
                                NDI = true;
                            end
                        else    % No error
                            NDI = true;
                        end
                        if NDI 
                            trBlk{Alg_mode} = randi([0 1],trBlkSizes{Alg_mode}(cwIdx),1);
                            if (Alg_mode == 1)
                                setTransportBlock(encodeDLSCH_alg1,trBlk{Alg_mode},cwIdx-1,harqProcIdx-1);
                            else
                                setTransportBlock(encodeDLSCH_alg2,trBlk{Alg_mode},cwIdx-1,harqProcIdx-1);
                            end
                        end
                    end

                    % Encode the DL-SCH transport blocks
                    if (Alg_mode == 1)
                        codedTrBlock = encodeDLSCH_alg1(pdsch_AlgSim(Alg_mode).Modulation,pdsch_AlgSim(Alg_mode).NLayers,...
                            pdschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
                    else
                        codedTrBlock = encodeDLSCH_alg2(pdsch_AlgSim(Alg_mode).Modulation,pdsch_AlgSim(Alg_mode).NLayers,...
                            pdschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
                    end

                    % Get wtx (precoding matrix) calculated in previous slot
                    wtx = newWtx;

                    % PDSCH modulation and precoding
                    pdschSymbols = nrPDSCH(codedTrBlock,pdsch_AlgSim(Alg_mode).Modulation,pdsch_AlgSim(Alg_mode).NLayers,gnb.NCellID,pdsch_AlgSim(Alg_mode).RNTI);
                    pdschSymbols = pdschSymbols*wtx;

                    % PDSCH mapping in grid associated with PDSCH transmission period
                    pdschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);
                    [~,pdschAntIndices] = nrExtractResources(pdschIndices,pdschGrid);
                    pdschGrid(pdschAntIndices) = pdschSymbols;

                    % PDSCH DM-RS precoding and mapping
                    for p = 1:size(dmrsSymbols,2)
                        [~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),pdschGrid);
                        pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols(:,p)*wtx(p,:);
                    end

                    % OFDM modulation of associated resource elements
                    txWaveform = hOFDMModulate(gnb, pdschGrid);

                    % Add the appropriate portion of SS burst waveform to the
                    % transmitted waveform
                    Nt = size(txWaveform,1);
                    txWaveform = txWaveform + ssbWaveform(ssbSampleIndex + (0:Nt-1),:);
                    ssbSampleIndex = mod(ssbSampleIndex + Nt,size(ssbWaveform,1));

                    % Pass data through channel model. Append zeros at the end of the
                    % transmitted waveform to flush channel content. These zeros take
                    % into account any delay introduced in the channel. This is a mix
                    % of multipath delay and implementation delay. This value may 
                    % change depending on the sampling rate, delay profile and delay
                    % spread
                    txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform,2))];
                    [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);

                    % Add AWGN to the received time domain waveform
                    % Normalize noise power to take account of sampling rate, which is
                    % a function of the IFFT size used in OFDM modulation. The SNR
                    % is defined per RE for each receive antenna (TS 38.101-4).
                    SNR = 10^(snrIn/20);    % Calculate linear noise gain
                    N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
                    noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
                    rxWaveform = rxWaveform + noise;

                    if (perfectChannelEstimator)
                        % Perfect synchronization. Use information provided by the channel
                        % to find the strongest multipath component
                        pathFilters = getPathFilters(channel); % get path filters for perfect channel estimation
                        [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
                    else
                        % Practical synchronization. Correlate the received waveform 
                        % with the PDSCH DM-RS to give timing offset estimate 't' and 
                        % correlation magnitude 'mag'. The function
                        % hSkipWeakTimingOffset is used to update the receiver timing
                        % offset. If the correlation peak in 'mag' is weak, the current
                        % timing estimate 't' is ignored and the previous estimate
                        % 'offset' is used
                        [t,mag] = nrTimingEstimate(rxWaveform,gnb.NRB,gnb.SubcarrierSpacing,pdsch_AlgSim(Alg_mode).NSlot,dmrsIndices,dmrsSymbols,'CyclicPrefix',gnb.CyclicPrefix); %#ok<UNRCH>
                        offset = hSkipWeakTimingOffset(offset,t,mag);
                    end
                    rxWaveform = rxWaveform(1+offset:end, :);

                    % Perform OFDM demodulation on the received data to recreate the
                    % resource grid, including padding in the event that practical
                    % synchronization results in an incomplete slot being demodulated
                    rxGrid = hOFDMDemodulate(gnb, rxWaveform);
                    [K,L,R] = size(rxGrid);
                    if (L < waveformInfo.SymbolsPerSlot)
                        rxGrid = cat(2,rxGrid,zeros(K,waveformInfo.SymbolsPerSlot-L,R));
                    end

                    if (perfectChannelEstimator)
                        % Perfect channel estimation, using the value of the path gains
                        % provided by the channel. This channel estimate does not
                        % include the effect of transmitter precoding
                        estChannelGrid = nrPerfectChannelEstimate(pathGains,pathFilters,gnb.NRB,gnb.SubcarrierSpacing,pdsch_AlgSim(Alg_mode).NSlot,offset,sampleTimes,gnb.CyclicPrefix);

                        % Get perfect noise estimate (from the noise realization)
                        noiseGrid = hOFDMDemodulate(gnb,noise(1+offset:end ,:));
                        noiseEst = var(noiseGrid(:));

                        % Get precoding matrix for next slot
                        newWtx = getPrecodingMatrix(pdsch_AlgSim(Alg_mode).PRBSet,pdsch_AlgSim(Alg_mode).NLayers,estChannelGrid);

                        % Apply precoding to estChannelGrid
                        estChannelGrid = precodeChannelEstimate(estChannelGrid,wtx.');
                    else
                        % Practical channel estimation between the received grid and
                        % each transmission layer, using the PDSCH DM-RS for each
                        % layer. This channel estimate includes the effect of
                        % transmitter precoding
                        [estChannelGrid,noiseEst] = nrChannelEstimate(rxGrid,dmrsIndices,dmrsSymbols,'CyclicPrefix',gnb.CyclicPrefix,'CDMLengths',pdschIndicesInfo.CDMLengths); %#ok<UNRCH>

                        % Remove precoding from estChannelGrid prior to precoding
                        % matrix calculation
                        estChannelGridPorts = precodeChannelEstimate(estChannelGrid,conj(wtx));

                        % Get precoding matrix for next slot
                        newWtx = getPrecodingMatrix(pdsch_AlgSim(Alg_mode).PRBSet,pdsch_AlgSim(Alg_mode).NLayers,estChannelGridPorts);
                    end

                    % Get PDSCH resource elements from the received grid
                    [pdschRx,pdschHest] = nrExtractResources(pdschIndices,rxGrid,estChannelGrid);

                    % Equalization
                    [pdschEq,csi] = nrEqualizeMMSE(pdschRx,pdschHest,noiseEst);

                    % Decode PDSCH physical channel
                    [dlschLLRs,rxSymbols] = nrPDSCHDecode(pdschEq,pdsch_AlgSim(Alg_mode).Modulation,gnb.NCellID,pdsch_AlgSim(Alg_mode).RNTI,noiseEst);

                    % Scale LLRs by CSI
                    csi = nrLayerDemap(csi); % CSI layer demapping
                    for cwIdx = 1:gnb.NumCW
                        Qm = length(dlschLLRs{cwIdx})/length(rxSymbols{cwIdx}); % bits per symbol
                        csi{cwIdx} = repmat(csi{cwIdx}.',Qm,1);   % expand by each bit per symbol
                        dlschLLRs{cwIdx} = dlschLLRs{cwIdx} .* csi{cwIdx}(:);   % scale
                    end

                    % Decode the DL-SCH transport channel
                    if (Alg_mode == 1)
                        decodeDLSCH_alg1.TransportBlockLength = trBlkSizes{Alg_mode};
                        [decbits,harqProcesses(harqProcIdx).blkerr] = decodeDLSCH_alg1(dlschLLRs,pdsch_AlgSim(Alg_mode).Modulation,pdsch_AlgSim(Alg_mode).NLayers,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
                    else
                        decodeDLSCH_alg2.TransportBlockLength = trBlkSizes{Alg_mode};
                        [decbits,harqProcesses(harqProcIdx).blkerr] = decodeDLSCH_alg2(dlschLLRs,pdsch_AlgSim(Alg_mode).Modulation,pdsch_AlgSim(Alg_mode).NLayers,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
                    end

                    % Store values to calculate throughput (only for active PDSCH instances)
                    if(any(trBlkSizes{Alg_mode} ~= 0))
                        bitTput{Alg_mode} = [bitTput{Alg_mode} trBlkSizes{Alg_mode}.*(1-harqProcesses(harqProcIdx).blkerr)];
                        txedTrBlkSizes{Alg_mode} = [txedTrBlkSizes{Alg_mode} trBlkSizes{Alg_mode}];                         
                    end

                    % Update starting symbol number of next PDSCH transmission
                    gnb.NSymbol = gnb.NSymbol + size(pdschGrid,2);
                    % Update count of overall number of PDSCH transmissions
                    pdsch_AlgSim(Alg_mode).NSlot = pdsch_AlgSim(Alg_mode).NSlot + 1;
                    % Update HARQ process counter
                    harqProcCntr = harqProcCntr + 1;

                    % Display transport block error information per codeword managed by current HARQ process
                    fprintf('\n(%3.2f%%) HARQ Proc %d: ',100*gnb.NSymbol/NSymbols,harqProcIdx);
                    estrings = {'passed','failed'};
                    rvi = harqProcesses(harqProcIdx).RVIdx; 
                    for cw=1:length(rvi)
                        cwrvi = rvi(cw);
                        % Create a report on the RV state given position in RV sequence and decoding error
                        if cwrvi == 1
                            ts = sprintf('Initial transmission (RV=%d)',rvSeq(cwrvi));
                        else
                            ts = sprintf('Retransmission #%d (RV=%d)',cwrvi-1,rvSeq(cwrvi));
                        end
                        fprintf('CW%d:%s %s. ',cw-1,ts,estrings{1+harqProcesses(harqProcIdx).blkerr(cw)}); 
                    end

                  end
                % Calculate maximum and simulated throughput
                maxThroughput(iter, Alg_mode) = sum(txedTrBlkSizes{Alg_mode}); % Max possible throughput
                simThroughput(iter, Alg_mode) = sum(bitTput{Alg_mode});      % Simulated throughput

              %% Implement DL AMC and update MCS
              if (harqProcesses(harqProcIdx).blkerr == 1) %% simulation based on iterations of 1 TTI, so BLER = 1 means NACK
                  AN = 0;
              else
                  AN = 1;
              end
              fprintf('\n Implementing DL AMC Algorithm %d.\n', Alg_mode);
              if (Alg_mode == 1)
                  [ModulationType, CodingRate, MCS_alg1] = DL_AMC_AN_alg1(MCS_table, MCS_alg1, AN);
                  pdsch_AlgSim(Alg_mode).Modulation = ModulationType;
                  pdsch_AlgSim(Alg_mode).TargetCodeRate = CodingRate;
                  encodeDLSCH_alg1.TargetCodeRate = pdsch_AlgSim(Alg_mode).TargetCodeRate;
                  decodeDLSCH_alg1.TargetCodeRate = pdsch_AlgSim(Alg_mode).TargetCodeRate;
              else
                  [ModulationType, CodingRate, MCS_alg2] = DL_AMC_AN_alg2(MCS_table, MCS_alg2, AN);
                  pdsch_AlgSim(Alg_mode).Modulation = ModulationType;
                  pdsch_AlgSim(Alg_mode).TargetCodeRate = CodingRate;
                  encodeDLSCH_alg2.TargetCodeRate = pdsch_AlgSim(Alg_mode).TargetCodeRate;
                  decodeDLSCH_alg2.TargetCodeRate = pdsch_AlgSim(Alg_mode).TargetCodeRate;
              end
    end
end
% clear DL_AMC_AN_alg1
% clear AN_Counter_alg1
% clear AN_Counter_alg2
%% plot results
% N_frames = Sim_iter*simParameters.nTTI/20;
% Throughput_percent = zeros(N_frames,1);
% frame_idx = 0;
% for nf = 1:N_frames
%     Throughput_percent(nf) = sum(simThroughput((nf-1)*20+1:nf*20))/sum(maxThroughput((nf-1)*20+1:nf*20));
% end
% figure;
% hold on
% plot(1:N_frames,Throughput_percent,'o-.')
% grid on
% hold off
Throughput_accumulate = zeros(Sim_iter,2);
BLER_accumulate = zeros(Sim_iter,2);
for iter = 1:Sim_iter
    for Alg_mode = 1:2
        Throughput_accumulate(iter, Alg_mode) = 1e-6*sum(simThroughput(1:iter, Alg_mode))/(iter*0.5*1e-3);
        BLER_accumulate(iter, Alg_mode) = (1 - sum(simThroughput(1:iter, Alg_mode))/sum(maxThroughput(1:iter, Alg_mode)))*100;
    end
end
figure();
hold on
plot(1:Sim_iter,BLER_accumulate(:,1),'.r-.')
plot(1:Sim_iter,BLER_accumulate(:,2),'.b-.')
xlabel('No. of Slots'); ylabel('BLER (%)');
title(sprintf('BLER Over Time at SNR %d dB, (%dx%d)',...
              SNR_mean,nTxAnts,nRxAnts));
grid on
hold off

figure();
hold on
plot(1:Sim_iter,Throughput_accumulate(:,1),'dr-.')
plot(1:Sim_iter,Throughput_accumulate(:,2),'db-.')
xlabel('No. of Slots'); ylabel('Throughput (Mbps)');
title(sprintf('Throughput Over Time at SNR %d dB, (%dx%d)',...
              SNR_mean,nTxAnts,nRxAnts));
grid on
hold off

figure();
hold on
plot(1:Sim_iter,MCSidx_record_alg1,'*r-.')
plot(1:Sim_iter,MCSidx_record_alg2,'*b-.')
xlabel('No. of Slots'); ylabel('MCS Index');
title(sprintf('MCS Adjustments Over Time at SNR %d dB, MCS Table %d',...
              SNR_mean,MCS_table));
grid on
hold off

%% save simulation results
cd Sim_results\Alg_Compare\20201108_2000iter_BLER10percent_startingMCS27_SNR14_median\
save BLER_accumulate.mat BLER_accumulate
save Throughput_accumulate.mat Throughput_accumulate
save MCSidx_record_alg1.mat MCSidx_record_alg1
save MCSidx_record_alg2.mat MCSidx_record_alg2
cd ..\..\..\

%% Assisting local functions
function estChannelGrid = getInitialChannelEstimate(gnb,nTxAnts,channel)
% Obtain channel estimate before first transmission. This can be used to
% obtain a precoding matrix for the first slot.

    ofdmInfo = hOFDMInfo(gnb);
    
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
    
    % Temporary waveform (only needed for the sizes)
    tmpWaveform = zeros((ofdmInfo.SamplesPerSubframe/ofdmInfo.SlotsPerSubframe)+maxChDelay,nTxAnts);
    
    % Filter through channel    
    [~,pathGains,sampleTimes] = channel(tmpWaveform);
    
    % Perfect timing synch    
    pathFilters = getPathFilters(channel);
    offset = nrPerfectTimingEstimate(pathGains,pathFilters);
    
    nslot = gnb.NSymbol/ofdmInfo.SymbolsPerSlot;
    
    % Perfect channel estimate
    estChannelGrid = nrPerfectChannelEstimate(pathGains,pathFilters,gnb.NRB,gnb.SubcarrierSpacing,nslot,offset,sampleTimes);
    
end

function wtx = getPrecodingMatrix(PRBSet,NLayers,hestGrid)
% Calculate precoding matrix given an allocation and a channel estimate
    
    % Allocated subcarrier indices
    allocSc = (1:12)' + 12*PRBSet(:).';
    allocSc = allocSc(:);
    
    % Average channel estimate
    [~,~,R,P] = size(hestGrid);
    estAllocGrid = hestGrid(allocSc,:,:,:);
    Hest = permute(mean(reshape(estAllocGrid,[],R,P)),[2 3 1]);
    
    % SVD decomposition
    [~,~,V] = svd(Hest);
    wtx = V(:,1:NLayers).';
    wtx = wtx / sqrt(NLayers); % Normalize by NLayers

end

function estChannelGrid = precodeChannelEstimate(estChannelGrid,W)
% Apply precoding matrix W to the last dimension of the channel estimate

    % Linearize 4D matrix and reshape after multiplication
    K = size(estChannelGrid,1);
    L = size(estChannelGrid,2);
    R = size(estChannelGrid,3);
    estChannelGrid = reshape(estChannelGrid,K*L*R,[]);
    estChannelGrid = estChannelGrid * W;
    estChannelGrid = reshape(estChannelGrid,K,L,R,[]);

end
function [mappedPRB,mappedSymbols] = mapNumerology(subcarriers,symbols,nrbs,nrbt,fs,ft)
% Map the SSBurst numerology to PDSCH numerology. The outputs are:
%   - mappedPRB: 0-based PRB indices for carrier resource grid (arranged in a column)
%   - mappedSymbols: 0-based OFDM symbol indices in a slot for carrier resource grid (arranged in a row)
%     carrier resource grid is sized using gnb.NRB, gnb.CyclicPrefix, spanning 1 slot
% The input parameters are:
%   - subcarriers: 1-based row subscripts for SSB resource grid (arranged in a column)
%   - symbols: 1-based column subscripts for SSB resource grid (arranged in an N-by-4 matrix, 4 symbols for each transmitted burst in a row, N transmitted bursts)
%     SSB resource grid is sized using ssbInfo.NRB, normal CP, spanning 5 subframes
%   - nrbs: source (SSB) NRB
%   - nrbt: target (carrier) NRB
%   - fs: source (SSB) SCS
%   - ft: target (carrier) SCS

    mappedPRB = unique(fix((subcarriers-(nrbs*6) - 1)*fs/(ft*12) + nrbt/2),'stable');
    
    symbols = symbols.';
    symbols = symbols(:).' - 1;

    if (ft < fs)
        % If ft/fs < 1, reduction
        mappedSymbols = unique(fix(symbols*ft/fs),'stable');
    else
        % Else, repetition by ft/fs
        mappedSymbols = reshape((0:(ft/fs-1))' + symbols(:)'*ft/fs,1,[]);
    end
    
end