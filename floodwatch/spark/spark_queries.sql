-- QUERY: events_per_year
WITH years AS (SELECT explode(sequence(2001,2025)) AS year)
SELECT y.year, count(e.event_id) AS n_events FROM years y
LEFT JOIN flood_events e ON y.year=e.year GROUP BY y.year ORDER BY y.year;

-- QUERY: tcwv_before_onset
SELECT 'three_days_before_onset' AS population, count(*) AS n_event_days,
       count(f.tcwv_pct) AS n_valid_days, avg(f.tcwv_pct) AS mean_tcwv_pct
FROM flood_events e JOIN features_daily f
ON f.date BETWEEN date_sub(cast(e.start AS DATE),3) AND date_sub(cast(e.start AS DATE),1)
UNION ALL
SELECT 'all_monsoon_days', count(*), count(tcwv_pct), avg(tcwv_pct)
FROM features_daily WHERE month(date) BETWEEN 6 AND 10;

-- QUERY: tcwv_by_event
WITH baseline AS (
 SELECT avg(tcwv_pct) AS all_monsoon_mean_tcwv_pct FROM features_daily
 WHERE month(date) BETWEEN 6 AND 10
)
SELECT e.event_id, e.start AS onset, e.peak_date, count(f.tcwv_pct) AS n_valid_days,
 avg(f.tcwv_pct) AS pre_onset_3d_mean_tcwv_pct, b.all_monsoon_mean_tcwv_pct
FROM flood_events e CROSS JOIN baseline b LEFT JOIN features_daily f
ON f.date BETWEEN date_sub(cast(e.start AS DATE),3) AND date_sub(cast(e.start AS DATE),1)
GROUP BY e.event_id,e.start,e.peak_date,b.all_monsoon_mean_tcwv_pct ORDER BY e.start;

-- QUERY: conditional_probabilities
WITH training AS (
 SELECT f.*, coalesce(d.flood,0) AS flood,
 CASE WHEN EXISTS (SELECT 1 FROM flood_events e WHERE cast(e.start AS DATE) > f.date AND cast(e.start AS DATE) <= date_add(f.date,1)) THEN 1 ELSE 0 END AS onset_1d,
 CASE WHEN EXISTS (SELECT 1 FROM flood_events e WHERE cast(e.start AS DATE) > f.date AND cast(e.start AS DATE) <= date_add(f.date,3)) THEN 1 ELSE 0 END AS onset_3d
 FROM features_daily f JOIN flood_daily d ON f.date=cast(d.date AS DATE)
 WHERE f.date BETWEEN DATE '2001-01-01' AND DATE '2019-12-31' AND month(f.date) BETWEEN 6 AND 10
), expanded AS (
 SELECT *, explode(array(1,3)) AS horizon_days FROM training
), populations AS (
 SELECT *, 'all_monsoon_days' AS population FROM expanded
 UNION ALL SELECT *, 'not_currently_flooding' AS population FROM expanded WHERE flood=0
), conditions AS (
 SELECT *, explode(array(
 named_struct('condition','all_days','eligible',true),
 named_struct('condition','tcwv_pct_ge_90','eligible',tcwv_pct>=90),
 named_struct('condition','tcwv_pct_ge_95','eligible',tcwv_pct>=95),
 named_struct('condition','tcwv_pct_ge_95_and_soil_gt_75','eligible',tcwv_pct>=95 AND sm_0_7_pct>75),
 named_struct('condition','rain_3d_pct_ge_95','eligible',rain_3d_pct>=95))) AS cond
 FROM populations
)
SELECT population, cond.condition AS condition, horizon_days, count(*) AS n_days,
 sum(CASE WHEN horizon_days=1 THEN onset_1d ELSE onset_3d END) AS onset_within_horizon_days,
 avg(CASE WHEN horizon_days=1 THEN onset_1d ELSE onset_3d END) AS probability
FROM conditions WHERE cond.eligible GROUP BY population,cond.condition,horizon_days
ORDER BY horizon_days,population,condition;
