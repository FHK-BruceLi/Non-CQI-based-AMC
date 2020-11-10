function [MCS_adjustment] = AN_Counter_alg2(AN, Threshold_ACK, Threshold_NACK)
% MCS_adjustment - 1: raise MCS by 1; -1: lower MCS by 1; 0: keep the
% current MCS
%% define local static varibles
persistent Total_ACK_alg2;
persistent Total_NACK_alg2;
% persistent Consecutive_ACK;
% persistent Last_AN;
persistent balance_counter;

if isempty(Total_ACK_alg2)
    Total_ACK_alg2 = 0;
end
if isempty(Total_NACK_alg2)
    Total_NACK_alg2 = 0;
end
% if isempty(Consecutive_ACK)
%     Consecutive_ACK = 0;
% end
% if isempty(Last_AN)
%     Last_AN = 0;
% end
if isempty(balance_counter)
    balance_counter = 0;
end
%% main algorithm
if (AN == 1)% ACK
    Total_ACK_alg2 = Total_ACK_alg2 + 1;
%     if (Last_AN == 0 || Last_AN == 1)
%         Consecutive_ACK = Consecutive_ACK + 1;
%     else
%         Consecutive_ACK = 0;
%     end 
%     Last_AN = 1;
    if (Total_ACK_alg2 == Threshold_ACK)
        balance_counter = 1;
    end

else
    Total_NACK_alg2 = Total_NACK_alg2 + 1;
    if (Total_NACK_alg2 == Threshold_NACK)
        balance_counter = -1;
    end
%     Last_AN = 2;
end

if (balance_counter > 0)
    MCS_adjustment = 1;
    Total_ACK_alg2 = Total_ACK_alg2 - Threshold_ACK;
    balance_counter = 0;
elseif (balance_counter < 0)
    MCS_adjustment = -1;
    Total_NACK_alg2 = Total_NACK_alg2 - Threshold_NACK;    
    balance_counter = 0;
else
    MCS_adjustment = 0;
end
end