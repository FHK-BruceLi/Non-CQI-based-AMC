function [NACK_Ratio, Consecutive_ACK, restart] = AN_Counter_alg1(AN, Counter_Size, Threshold_NACK, Forgetting_factor, Restart, NACK_Ratio_weighted)

%% define local static varibles
persistent Total_NACK;
persistent NACK_R;
persistent n;
persistent Last_AN;
persistent Consecutive_A;
persistent Consecutive_N;
if isempty(Total_NACK)
    Total_NACK = 0;
end
if isempty(NACK_R)
    NACK_R = 0;
end
if isempty(n)
    n = 0;
end
if isempty(Last_AN)
    Last_AN = 0;
end
if isempty(Consecutive_A)
    Consecutive_A = 0;
end
if isempty(Consecutive_N)
    Consecutive_N = 0;
end
%% main algorithm
if (Restart==1)
    Total_NACK = 0;
    NACK_R = 0;
    n = 0;
    Last_AN = 0;
    Consecutive_A = 0;
    Consecutive_N = 0;
end

if (AN == 1) %% ACK
    if (Last_AN == 0 || Last_AN == 1)
        Consecutive_A = Consecutive_A + 1;
    else
        Consecutive_A = 0;
    end 
    Last_AN = 1;
    if (n < Counter_Size)
        n = n + 1;
    else %% (n = Counter_Size)
        if (NACK_R == 0)
            n = 0;
        else
            NACK_R = Total_NACK/n;
            NACK_R = Forgetting_factor*NACK_Ratio_weighted + (1 - Forgetting_factor)*NACK_R;                
        end
    end
else
    Total_NACK = Total_NACK + 1;
    if (Last_AN == 0 || Last_AN == 2)
        Consecutive_N = Consecutive_N + 1;
    else
        Consecutive_N = 0;
    end

    if (Consecutive_N == Threshold_NACK)
        NACK_R = 1;
    end
    Last_AN = 2;  
    if (n < Counter_Size)
        n = n + 1;
    else %% (n = Counter_Size)
        NACK_R = Total_NACK/n;
        NACK_R = Forgetting_factor*NACK_Ratio_weighted + (1 - Forgetting_factor)*NACK_R;
    end
end
%     if (n < Counter_Size)
%         n = n +1;
%         if (AN == 1) %% ACK
%             if (Last_AN == 0 || Last_AN == 1)
%                 Consecutive_A = Consecutive_A + 1;
%             else
%                 Consecutive_A = 0;
%             end
%             Last_AN = 1;
%         else %% NACK
%             Total_NACK = Total_NACK + 1;
%             if (Last_AN == 0 || Last_AN == 2)
%                 Consecutive_N = Consecutive_N + 1;
%             else
%                 Consecutive_N = 0;
%             end
%             
%             if (Consecutive_N == Threshold_NACK)
%                 NACK_R = 1;
%             end
%             Last_AN = 2;
%         end
%     else %% (n = Counter_Size)
%         NACK_R = Total_NACK/n;
%         if (NACK_R > 0)
%             NACK_R = Forgetting_factor*NACK_Ratio_weighted + (1 - Forgetting_factor)*NACK_R;
%         else
%             n = 0;
%         end
%     end
NACK_Ratio = NACK_R;
Consecutive_ACK = Consecutive_A;
restart = 0;

end