WITH episodes AS (
  SELECT *
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/training_episode_metrics.csv')
)
SELECT episode,
       '10-episode mean reward' AS metric,
       rolling10_mean_R AS value,
       mean_R AS raw_mean_R,
       online_em AS raw_online_em,
       sf_recall AS raw_sf_recall,
       retrievals AS raw_retrievals,
       kl AS raw_kl
FROM episodes
UNION ALL
SELECT episode, '10-episode online EM', rolling10_online_em,
       mean_R, online_em, sf_recall, retrievals, kl
FROM episodes
UNION ALL
SELECT episode, '10-episode online SF recall', rolling10_sf_recall,
       mean_R, online_em, sf_recall, retrievals, kl
FROM episodes
ORDER BY episode, metric;
