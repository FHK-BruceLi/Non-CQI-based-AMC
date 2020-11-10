function [ModulationType, CodingRate, MCS_idx] = DL_AMC_AN_alg2(MCS_table, MCS_idx, AN)


%% parameters
Threshold_ACK = 10;
Threshold_NACK = 1;
%% main
[MCS_adjustment] = AN_Counter_alg2(AN, Threshold_ACK, Threshold_NACK);
if (MCS_adjustment == 1)
    if (MCS_table == 1)
        if (MCS_idx < 28)
            MCS_idx = MCS_idx + 1;
        end
    elseif (MCS_table == 2)
        if (MCS_idx < 27)
            MCS_idx = MCS_idx + 1;
        end
    end
    fprintf('\n Raise MCS by 1 to %d.\n', MCS_idx);
elseif (MCS_adjustment == -1)
        if (MCS_idx > 0)
            MCS_idx = MCS_idx - 1;
        end
        fprintf('\n Lower MCS by 1 to %d.\n', MCS_idx);
else
    fprintf('\n Keep the current MCS %d.\n', MCS_idx);
end

[ModulationType, CodingRate] = MCSidx2MCS(MCS_table,MCS_idx);
end