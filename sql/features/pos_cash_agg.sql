SELECT
    p.SK_ID_CURR,
    
    -- Volume
    COUNT(DISTINCT p.SK_ID_PREV)                                    as pos_loan_count,
    COUNT(*)                                                         as pos_total_months,
    
    -- Instalment completion
    AVG(p.CNT_INSTALMENT)                                           as pos_avg_instalment,
    AVG(p.CNT_INSTALMENT_FUTURE)                                    as pos_avg_instalment_future,
    
    -- Late payment behaviour (most important)
    MAX(p.SK_DPD)                                                   as pos_max_dpd,
    AVG(p.SK_DPD)                                                   as pos_avg_dpd,
    SUM(CASE WHEN p.SK_DPD > 0 THEN 1 ELSE 0 END)                 as pos_late_payment_count,
    SUM(CASE WHEN p.SK_DPD > 30 THEN 1 ELSE 0 END)                as pos_dpd_over_30_count,
    SUM(CASE WHEN p.SK_DPD > 60 THEN 1 ELSE 0 END)                as pos_dpd_over_60_count,
    
    -- Ratio: late payments / total months
    ROUND(
        SUM(CASE WHEN p.SK_DPD > 0 THEN 1 ELSE 0 END) * 1.0 
        / COUNT(*), 4
    )                                                               as pos_late_payment_rate,
    
    -- Contract status counts
    SUM(CASE WHEN p.NAME_CONTRACT_STATUS = 'Active' THEN 1 ELSE 0 END)    as pos_active_count,
    SUM(CASE WHEN p.NAME_CONTRACT_STATUS = 'Completed' THEN 1 ELSE 0 END) as pos_completed_count

FROM read_csv_auto('C:/Users/BeratErcan/Desktop/Data/raw/POS_CASH_balance.csv') p
GROUP BY p.SK_ID_CURR