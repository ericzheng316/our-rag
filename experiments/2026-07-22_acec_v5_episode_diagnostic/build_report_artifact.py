#!/usr/bin/env python3
"""Build the canonical Data Analytics report artifact for the diagnostic."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(name: str):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value)


def build_artifact():
    summary = json.loads((ROOT / "analysis_results.json").read_text(encoding="utf-8"))
    drivers = read_csv("question_type_drivers.csv")
    batches = read_csv("batch_diagnostic.csv")
    transitions = read_csv("transition_examples.csv")
    training = read_csv("training_episode_metrics.csv")
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")

    checkpoint_labels = {"step50": "Episode 50", "step75": "Episode 75", "step100": "Episode 100"}
    checkpoint_rows = []
    checkpoint_long = []
    for variant in ("step50", "step75", "step100"):
        values = summary["overall"][variant]
        row = {
            "checkpoint": checkpoint_labels[variant],
            "episode": int(variant.removeprefix("step")),
            **{
                key: number(values[key])
                for key in (
                    "direct_em",
                    "direct_f1",
                    "processed_em",
                    "processed_f1",
                    "gold_sf_recall",
                    "retrieval_calls",
                    "mean_prediction_words",
                )
            },
        }
        checkpoint_rows.append(row)
        for metric, label in (("processed_em", "Processed EM"), ("processed_f1", "Processed F1")):
            checkpoint_long.append(
                {
                    **row,
                    "metric": label,
                    "value": row[metric],
                }
            )

    comparison_100 = summary["comparisons"]["step50_to_step100"]
    headline = [{
        "best_episode": 50,
        "step100_vs_step50_processed_f1_delta": comparison_100["processed_f1_delta"],
        "step100_vs_step50_gold_sf_delta": (
            summary["overall"]["step100"]["gold_sf_recall"]
            - summary["overall"]["step50"]["gold_sf_recall"]
        ),
        "step100_vs_step50_retrieval_delta": (
            summary["overall"]["step100"]["retrieval_calls"]
            - summary["overall"]["step50"]["retrieval_calls"]
        ),
        "heldout_questions": 256,
    }]

    training_long = []
    for row in training:
        context = {
            "episode": int(row["episode"]),
            "raw_mean_R": number(row["mean_R"]),
            "raw_online_em": number(row["online_em"]),
            "raw_sf_recall": number(row["sf_recall"]),
            "raw_retrievals": number(row["retrievals"]),
            "raw_kl": number(row["kl"]),
        }
        for field, label in (
            ("rolling10_mean_R", "10-episode mean reward"),
            ("rolling10_online_em", "10-episode online EM"),
            ("rolling10_sf_recall", "10-episode online SF recall"),
        ):
            training_long.append({**context, "metric": label, "value": number(row[field])})

    driver_rows = []
    for row in drivers:
        driver_rows.append({
            "question_type": row["question_type"],
            "n": int(row["n"]),
            "step50_processed_f1": number(row["step50_processed_f1"]),
            "step100_processed_f1": number(row["step100_processed_f1"]),
            "f1_delta": number(row["step100_vs_step50_f1_delta"]),
            "f1_contribution": number(row["step100_vs_step50_f1_contribution"]),
            "em_delta": number(row["step100_vs_step50_em_delta"]),
        })

    batch_lookup = {
        (row["variant"], int(row["batch"])): row
        for row in batches
        if row["variant"] in {"step50", "step100"}
    }
    batch_scatter = []
    for batch in range(1, 17):
        before = batch_lookup[("step50", batch)]
        after = batch_lookup[("step100", batch)]
        batch_scatter.append({
            "batch": f"Batch {batch}",
            "batch_number": batch,
            "question_count": 16,
            "retrieval_delta": number(after["retrieval_calls"]) - number(before["retrieval_calls"]),
            "gold_sf_delta": number(after["gold_sf_recall"]) - number(before["gold_sf_recall"]),
            "processed_f1_delta": number(after["processed_f1"]) - number(before["processed_f1"]),
            "step50_retrieval_calls": number(before["retrieval_calls"]),
            "step100_retrieval_calls": number(after["retrieval_calls"]),
            "step50_gold_sf_recall": number(before["gold_sf_recall"]),
            "step100_gold_sf_recall": number(after["gold_sf_recall"]),
        })

    loss_rows = []
    for row in transitions:
        pattern = row["pattern_50_75_100"]
        if pattern.startswith("1") and pattern.endswith("0"):
            loss_rows.append({
                "id": row["id"],
                "question_type": row["question_type"],
                "question": row["question"],
                "gold": row["golden_answers"],
                "step50_answer": row["step50_prediction"],
                "step100_answer": row["step100_prediction"],
                "step100_f1": number(row["step100_processed_f1"]),
                "pattern": pattern,
            })

    source_common = {
        "engine": "DuckDB",
        "language": "sql",
        "executed_at": generated_at,
        "filters": [
            "Fixed 256-question HotpotQA dev sample selected with seed 20260721",
            "Temperature 0, one rollout per checkpoint and question",
            "Checkpoint comparison limited to episodes 50, 75, and 100",
        ],
    }
    sources = [
        {
            "id": "diagnostic_summary",
            "label": "ACEC checkpoint diagnostic summary",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/analysis_results.json",
            "query": {
                **source_common,
                "description": "Deterministic alignment, paired bootstrap, stability, and metric decomposition generated by analyze_episode_checkpoints.py.",
                "tables_used": [
                    "analysis_results.json",
                    "per_question_diagnostic.csv",
                    "training_episode_metrics.csv",
                    "batch_diagnostic.csv",
                ],
                "metric_definitions": [
                    "Processed EM/F1: maximum normalized HotpotQA metric over the original response and R3-extracted answer candidates.",
                    "Checkpoint deltas: paired after-minus-step50 mean over the same 256 question ids.",
                    "Confidence intervals: 10,000 paired bootstrap resamples over question ids.",
                ],
            },
        },
        {
            "id": "headline_query",
            "label": "Headline checkpoint deltas",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/headline.sql",
            "query": {
                **source_common,
                "description": "Recomputes the recommended checkpoint and Episode-100-minus-50 headline deltas from the aligned per-question and batch tables.",
                "tables_used": ["per_question_diagnostic.csv", "batch_diagnostic.csv"],
                "metric_definitions": [
                    "Headline deltas are paired or aggregate after-minus-before differences on the fixed held-out sample.",
                ],
            },
        },
        {
            "id": "checkpoint_query",
            "label": "Checkpoint metric summary",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/checkpoint_metrics.sql",
            "query": {
                **source_common,
                "description": "Aggregates direct and processed answer quality per checkpoint, then joins mean held-out Gold-SF recall and retrieval calls.",
                "tables_used": ["per_question_diagnostic.csv", "batch_diagnostic.csv"],
                "metric_definitions": [
                    "Answer metrics are means over the same 256 question ids.",
                    "Gold-SF recall and retrieval calls are means of 16 equal-sized evaluation batches.",
                ],
            },
        },
        {
            "id": "checkpoint_long_query",
            "label": "Long-form checkpoint answer metrics",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/checkpoint_metrics_long.sql",
            "query": {
                **source_common,
                "description": "Unpivots processed EM and processed F1 for the checkpoint comparison chart while retaining tooltip context.",
                "tables_used": ["per_question_diagnostic.csv", "batch_diagnostic.csv"],
                "metric_definitions": [
                    "Processed EM/F1 are plotted as separate rows for each checkpoint.",
                ],
            },
        },
        {
            "id": "question_driver_query",
            "label": "Question-type contribution decomposition",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/question_type_drivers.sql",
            "query": {
                **source_common,
                "description": "Returns lexical question-type counts, within-type deltas, and additive contributions to the total Step-100-minus-50 F1 change.",
                "tables_used": ["question_type_drivers.csv"],
                "metric_definitions": [
                    "Question type is a deterministic lexical heuristic, not an official HotpotQA field.",
                    "Contributions sum to the overall processed-F1 delta, subject to floating-point precision.",
                ],
            },
        },
        {
            "id": "loss_query",
            "label": "Episode-50 correct to Episode-100 wrong transitions",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/step100_losses.sql",
            "query": {
                **source_common,
                "description": "Selects aligned questions whose processed EM changed from correct at Episode 50 to wrong at Episode 100.",
                "tables_used": ["transition_examples.csv"],
                "metric_definitions": [
                    "Exact loss: Step-50 processed EM equals 1 and Step-100 processed EM equals 0.",
                ],
            },
        },
        {
            "id": "batch_query",
            "label": "Held-out retrieval and Gold-SF batch deltas",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/batch_scatter.sql",
            "query": {
                **source_common,
                "description": "Joins equal-sized Episode-50 and Episode-100 evaluation batches to compute retrieval, Gold-SF, and processed-F1 deltas.",
                "tables_used": ["batch_diagnostic.csv"],
                "metric_definitions": [
                    "Each point aggregates 16 held-out questions; correlations are descriptive and not per-question causal estimates.",
                ],
            },
        },
        {
            "id": "training_query",
            "label": "ACEC-v5 trailing online training metrics",
            "path": "experiments/2026-07-22_acec_v5_episode_diagnostic/queries/training_trend.sql",
            "query": {
                **source_common,
                "description": "Unpivots trailing 10-episode reward, online EM, and online SF recall while retaining raw episode retrieval and KL context.",
                "tables_used": ["training_episode_metrics.csv"],
                "filters": [
                    "Episodes 1-100, 16 sequential training questions x 8 rollouts per episode",
                    "Trailing averages use up to 10 completed episodes and no look-ahead",
                ],
                "metric_definitions": [
                    "Online metrics use changing sequential training questions and are not fixed validation metrics.",
                    "Mean reward uses answer 1.0, ACEC coverage 0.3, format penalty 0.1, and retrieval cost 0.05.",
                ],
            },
        },
    ]

    title = "ACEC-v5：为什么 Episode 50 好于 75/100"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "固定 HotpotQA held-out 与训练日志的 checkpoint 退化诊断。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "best_checkpoint",
                "description": "当前固定 held-out 上 processed EM/F1 与 Gold-SF recall 的最佳综合 checkpoint。",
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [{"label": "推荐 checkpoint", "field": "best_episode", "format": "number"}],
            },
            {
                "id": "f1_change",
                "description": "Episode 100 相对 Episode 50 的配对 processed-F1 变化。",
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [{"label": "Processed F1 变化", "field": "step100_vs_step50_processed_f1_delta", "format": "percent", "signed": True}],
            },
            {
                "id": "sf_change",
                "description": "Episode 100 相对 Episode 50 的固定 held-out Gold supporting-fact recall 变化。",
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [{"label": "Gold-SF recall 变化", "field": "step100_vs_step50_gold_sf_delta", "format": "percent", "signed": True}],
            },
            {
                "id": "retrieval_change",
                "description": "Episode 100 相对 Episode 50 的每题平均检索调用变化。",
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [{"label": "每题检索变化", "field": "step100_vs_step50_retrieval_delta", "format": "number", "signed": True}],
            },
        ],
        "charts": [
            {
                "id": "checkpoint_answer_metrics",
                "title": "Held-out answer metrics at saved checkpoints",
                "subtitle": "Episode 50 leads both processed EM and processed F1 on the fixed 256-question sample.",
                "type": "bar",
                "dataset": "checkpoint_long",
                "sourceId": "checkpoint_long_query",
                "encodings": {
                    "x": {"field": "checkpoint", "type": "ordinal", "label": "Checkpoint"},
                    "y": {"field": "value", "type": "quantitative", "label": "Score", "format": "percent"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                    "tooltip": [
                        {"field": "direct_em", "type": "quantitative", "label": "Direct EM", "format": "percent"},
                        {"field": "gold_sf_recall", "type": "quantitative", "label": "Gold-SF recall", "format": "percent"},
                        {"field": "retrieval_calls", "type": "quantitative", "label": "Retrieval calls"},
                    ],
                },
                "yAxisTitle": "Score",
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "question_type_contribution",
                "title": "Step100-minus-step50 F1 contribution by question type",
                "subtitle": "Who and what questions drive more negative F1 than the net decline; small slices are directional only.",
                "type": "bar",
                "dataset": "question_drivers",
                "sourceId": "question_driver_query",
                "encodings": {
                    "x": {"field": "question_type", "type": "nominal", "label": "Question type"},
                    "y": {"field": "f1_contribution", "type": "quantitative", "label": "Contribution", "format": "percent"},
                    "tooltip": [
                        {"field": "n", "type": "quantitative", "label": "Questions"},
                        {"field": "f1_delta", "type": "quantitative", "label": "Within-type F1 delta", "format": "percent"},
                        {"field": "em_delta", "type": "quantitative", "label": "Within-type EM delta", "format": "percent"},
                    ],
                },
                "yAxisTitle": "Contribution to overall F1 change",
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "retrieval_sf_scatter",
                "title": "Batch-level retrieval and Gold-SF changes from Episode 50 to 100",
                "subtitle": "Each point is 16 questions; 11 of 16 batches reduce both retrieval calls and Gold-SF recall.",
                "type": "scatter",
                "dataset": "batch_scatter",
                "sourceId": "batch_query",
                "encodings": {
                    "x": {"field": "retrieval_delta", "type": "quantitative", "label": "Retrieval-call delta"},
                    "y": {"field": "gold_sf_delta", "type": "quantitative", "label": "Gold-SF recall delta", "format": "percent"},
                    "label": {"field": "batch", "type": "text", "label": "Batch"},
                    "tooltip": [
                        {"field": "processed_f1_delta", "type": "quantitative", "label": "Processed-F1 delta", "format": "percent"},
                        {"field": "question_count", "type": "quantitative", "label": "Questions"},
                    ],
                },
                "xAxisTitle": "Change in retrieval calls per question",
                "yAxisTitle": "Change in Gold-SF recall",
                "layout": "full",
            },
            {
                "id": "training_proxy_trend",
                "title": "Trailing 10-episode online training metrics",
                "subtitle": "Online reward and EM recover by Episode 100, but each point uses changing sequential training questions.",
                "type": "line",
                "dataset": "training_long",
                "sourceId": "training_query",
                "encodings": {
                    "x": {"field": "episode", "type": "quantitative", "label": "Episode"},
                    "y": {"field": "value", "type": "quantitative", "label": "10-episode mean"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                    "tooltip": [
                        {"field": "raw_retrievals", "type": "quantitative", "label": "Episode retrieval calls"},
                        {"field": "raw_kl", "type": "quantitative", "label": "Episode KL"},
                    ],
                },
                "xAxisTitle": "Episode",
                "yAxisTitle": "Trailing 10-episode mean",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "checkpoint_summary",
                "title": "Checkpoint metric summary",
                "subtitle": "Same 256 questions, temperature 0, one rollout per checkpoint.",
                "dataset": "checkpoint_metrics",
                "sourceId": "checkpoint_query",
                "defaultSort": {"field": "episode", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
                    {"field": "episode", "label": "Episode", "type": "number"},
                    {"field": "processed_em", "label": "Processed EM", "format": "percent"},
                    {"field": "processed_f1", "label": "Processed F1", "format": "percent"},
                    {"field": "direct_f1", "label": "Direct F1", "format": "percent"},
                    {"field": "gold_sf_recall", "label": "Gold-SF recall", "format": "percent"},
                    {"field": "retrieval_calls", "label": "Retrieval calls", "format": "number"},
                ],
            },
            {
                "id": "step100_losses",
                "title": "Questions correct at Episode 50 and wrong at Episode 100",
                "subtitle": "17 exact losses; rows with zero Step-100 F1 represent complete answer replacement.",
                "dataset": "step100_losses",
                "sourceId": "loss_query",
                "defaultSort": {"field": "step100_f1", "direction": "asc"},
                "density": "comfortable",
                "layout": "full",
                "columns": [
                    {"field": "id", "label": "ID", "type": "text"},
                    {"field": "question_type", "label": "Type", "type": "text"},
                    {"field": "question", "label": "Question", "type": "text"},
                    {"field": "gold", "label": "Gold", "type": "text"},
                    {"field": "step50_answer", "label": "Episode 50", "type": "text"},
                    {"field": "step100_answer", "label": "Episode 100", "type": "text"},
                    {"field": "step100_f1", "label": "Episode-100 F1", "format": "percent"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "diagnostic_summary",
                "body": "## 技术结论\n\n**当前不应把 ACEC-v5 原样延长到 500/800 episode。** Episode 50 是现有 checkpoint 中最合理的停止点。到 Episode 100，processed EM 下降 3.13 个百分点、processed F1 下降 3.27 个百分点、Gold-SF recall 下降 5.08 个百分点，同时每题检索又减少 0.164 次。后期策略确实更省检索，但固定 held-out 的证据覆盖和答案质量没有保持。",
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["best_checkpoint", "f1_change", "sf_change", "retrieval_change"]},
            {
                "id": "checkpoint_finding",
                "type": "markdown",
                "sourceId": "diagnostic_summary",
                "body": "## 固定 held-out 的多项指标同时在 Episode 50 达峰\n\nProcessed、direct 与 substring 指标给出同一方向，而不是单一 extractor 口径造成的假象。Episode 75 相比 50 有 20 个 exact loss、10 个 win；Episode 100 有 17 个 loss、9 个 win。两组配对置信区间仍跨 0，因此 256 题不足以证明 step50 在统计上必胜，但当前点估计一致反对继续训练后自动变好。",
            },
            {"id": "checkpoint_chart", "type": "chart", "chartId": "checkpoint_answer_metrics", "layout": "full"},
            {"id": "checkpoint_table", "type": "table", "tableId": "checkpoint_summary", "layout": "full"},
            {
                "id": "answer_regression",
                "type": "markdown",
                "sourceId": "loss_query",
                "body": "## 后期退化主要是正确实体被替换，而不是答案被截短\n\nEpisode 50 到 100 的 17 个 exact loss 中，11 个在 Episode 100 变成零 F1；平均答案长度反而从 3.23 增到 3.42 个词。`who` 与 `what` 两类合计贡献约 -3.57 个百分点 F1，超过总净下降 -3.27 点，`which` 和 numeric 的改善提供了部分抵消。73 题三版始终正确、142 题始终错误，只有 41 题决定 checkpoint 排名。",
            },
            {"id": "driver_chart", "type": "chart", "chartId": "question_type_contribution", "layout": "full"},
            {"id": "loss_table", "type": "table", "tableId": "step100_losses", "layout": "full"},
            {
                "id": "retrieval_compression",
                "type": "markdown",
                "sourceId": "batch_query",
                "body": "## 检索压缩与 Gold-SF 下降同步，但不能单独解释答案退化\n\nEpisode 50 到 100 有 14/16 个 held-out 批次减少检索，11/16 同时降低 Gold-SF recall；两种变化的批次相关系数为 0.58。可是 Gold-SF 变化与 processed-F1 变化几乎不相关。由于只有 16 个批次且没有逐题检索轨迹，最稳妥的解释是：**过早停止很可能损失证据，但 answer generator 的实体选择漂移也是独立问题。**",
            },
            {"id": "retrieval_chart", "type": "chart", "chartId": "retrieval_sf_scatter", "layout": "full"},
            {
                "id": "proxy_divergence",
                "type": "markdown",
                "sourceId": "training_query",
                "body": "## 在线训练代理与固定 held-out 在后期脱钩\n\n结束于 Episode 100 的最后 10 轮平均 meanR、online EM 和 online SF recall 都高于结束于 Episode 50 的窗口，KL 从约 0.063 升到 0.096，检索继续减少。可这些窗口使用不同的顺序训练问题，难度构成不断变化，所以它们不能充当验证集。这个结果支持“代理指标/泛化失配”，但单凭时间趋势还不能声称是因果性的 reward hacking。",
            },
            {"id": "training_chart", "type": "chart", "chartId": "training_proxy_trend", "layout": "full"},
            {
                "id": "methods_limits",
                "type": "markdown",
                "sourceId": "diagnostic_summary",
                "body": "## 口径、方法与鲁棒性\n\n分析固定对齐 256 个 question id，并独立重算 direct/processed EM/F1、配对转移与 10,000 次 bootstrap。检索只能分析到 16 题批次粒度。另一个可修复的测量噪声是：914 次 extractor 调用中，有 282 次对应已出现过的相同 `(question, prediction)`；2 个重复组给出了不同 processed EM，5 个组给出了不同 F1。跨 variant 缓存相同输入可以同时消除噪声并降低约 31% 的 extractor 调用。该噪声不会推翻 Episode 50 的结论。",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "body": "## 下一版应同时保护检索停止策略与答案生成器\n\n1. 暂定 Episode 50 为 v5 checkpoint，不原样继续到 500/800。\n2. 新 evaluator 保存逐题 action、query、document id/title、Gold-SF hit 和停止轮次。\n3. Processed evaluator 按 `(question, prediction)` 跨 variant 缓存。\n4. v6 为 ANSWER/停止决策增加 evidence-coverage guard；同时用更强或自适应 KL、answer-token loss mask，或少量高质量 SFT rehearsal 防止实体选择漂移。\n5. 下一次每 25 episode 用固定 held-out 的 processed EM/F1、semantic ACC、Gold-SF recall 和 retrieval calls 早停；只有答案质量不退化且效率改善时才扩展到 500。",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": "## 仍需回答的问题\n\n- 17 个 exact loss 中，哪些是过早停止，哪些是错误证据，哪些是拿到正确证据后仍答错？\n- 是否可以只训练检索/action token，而对 final-answer token 使用冻结或 SFT anchor？\n- Step50 的优势能否在第二个固定样本或更大 dev 集上复现？",
            },
        ],
    }

    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "checkpoint_metrics": checkpoint_rows,
            "checkpoint_long": checkpoint_long,
            "training_long": training_long,
            "question_drivers": driver_rows,
            "batch_scatter": batch_scatter,
            "step100_losses": loss_rows,
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


if __name__ == "__main__":
    output = ROOT / "report_artifact.json"
    output.write_text(json.dumps(build_artifact(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
