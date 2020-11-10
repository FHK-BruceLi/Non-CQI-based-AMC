function [ModulationType, CodingRate, MCS_idx] = DL_AMC_AN_alg1(MCS_table, MCS_idx, AN)

%% define local static varibles
persistent restart;
persistent NACK_Ratio_weighted;
if isempty(restart)
    restart = 0;
end
if isempty(NACK_Ratio_weighted)
    NACK_Ratio_weighted = 0;
end
%% parameters
Counter_Size = 10;
Threshold_NACK = 1;
Threshold_ACK = 10;
BLER_target = 0.1;
Forgetting_factor = 0.5;

%% main
[NACK_Ratio, Consecutive_ACK, Restart] = AN_Counter_alg1(AN, Counter_Size, Threshold_NACK, Forgetting_factor, restart, NACK_Ratio_weighted);
NACK_Ratio_weighted = NACK_Ratio;
restart = Restart;
if (NACK_Ratio == 0)
    if (Consecutive_ACK == Threshold_ACK)
        if (MCS_table == 1)
            if (MCS_idx < 28)
                MCS_idx = MCS_idx + 1;
            end
        elseif (MCS_table == 2)
            if (MCS_idx < 27)
                MCS_idx = MCS_idx + 1;
            end
        end
        restart = 1;
        fprintf('\n Consecutive ACK reaches Threshold, raise MCS by 1 to %d and restart A/N Counter.\n', MCS_idx);
    end
elseif (NACK_Ratio > BLER_target)
        if (MCS_idx > 0)
            MCS_idx = MCS_idx - 1;
        end
        restart = 1;
        fprintf('\n NACK_Ratio is %d, lower MCS by 1 to %d and restart A/N Counter.\n', NACK_Ratio, MCS_idx);
else
    restart = 1;
    fprintf('\n Keep the current MCS %d and restart A/N Counter.\n', MCS_idx);
end

[ModulationType, CodingRate] = MCSidx2MCS(MCS_table,MCS_idx);
end