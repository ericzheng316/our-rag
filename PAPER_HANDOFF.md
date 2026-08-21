# 论文交接清单(实验窗口 → 写作窗口,2026-08-19)

所有数字均为定稿值,预测文件在列出的路径下可直接复算。
口径:MuSiQue-Ans 全量 dev 2417,闭池=官方20段,alias-EM,贪心(另注明除外)。

## 1. 主结果与阶梯(同底座 Qwen3.5-9B)

| 行 | EM | F1 | 预测文件 |
|---|---|---|---|
| 闭卷 | 4.8 | — | logs/eval_a_ladder_*/a1_closedbook.jsonl |
| 单发RAG top10(开卷) | 15.4 | 24.9 | 同上 a2_singleshot.jsonl |
| 协议零样本(开卷) | 29.0 | 38.2 | 同上 a3_zeroshot.jsonl(格式错13.5%) |
| SFT v2.2(分层360) | 21.4* | — | logs/eval_v22_20260808T194734Z |
| SFT v3 | 34.5 | 40.7 | logs/eval_sftv3_fulldev_* |
| RL ep100 三种子 | 48.20/48.53/45.68 = 47.5±1.6 | 0.566 | eval_r7s100_fulldev_20260811T142930Z / eval_seed1_* / eval_seed2_* |
| **RL ep120 旗舰** | **50.27** | **0.593** | eval_sweep_20260814T104838Z/hops_s20_* |
| ep120 开卷(IRCoT 139,416) | 45.63 | 0.565 | eval_ob_ep120_* |
| v3-RL ep100 | 46.75 | 0.529 | eval_v3rl_fulldev_* |
*分层360口径,与全量不直接比

训练长度曲线(全量): ep100 48.20 / ep120 50.27 / ep140 47.58 / ep160 48.70 / ep180 48.53
ep120 vs ep100 配对 McNemar: 全量 z=2.99 (165:115), 4hop z=3.61 (44:16)

## 2. 塑形消融(核心因果)

无塑形 ep100: 46.55/0.539。塑形贡献: 全量+1.65 (z=2.37), 4hop+7.2 (z=3.98, 41:12),
难度梯度 E−2.2 / M+1.8 / H+5.3 (H桶 z=2.72)。文件: eval_noshape_fulldev_*

## 3. 迁移矩阵(零样本,闭池)

| | SFT v2.2 | RL ep120 | 增量 |
|---|---|---|---|
| HotpotQA (1200) | 37.5 | 39.3 | +1.8 |
| — bridge (960) | 35.4 | 34.7 | −0.7 |
| — comparison (240) | 45.8 | 57.5 | **+11.7** |
| 2Wiki (2400) | 47.2 | 56.1 | **+8.9** |
文件: eval_transfer_sftv22_* / eval_transfer_ep120_*
主张措辞: 多槽覆盖纪律零样本迁移, 增量精确落于需要该纪律的题型。

## 4. 技能/套路分解(难度桶=裸查gold覆盖, buckets 文件
/scratch/boyuz5/acec/musique_openbook/difficulty_buckets.json)

SFT→RL120 开卷: E+2.0 / M+6.8 / H+6.6, 且检索次数全桶下降
RL100→RL120: 闭池 E+2.0/M+1.5/H+4.5; 开卷 E+0.5/M−0.8/H−1.6 → 峰值=池套路

## 5. 基线

R3-RAG 同栈实测(IRCoT语料): direct 6.6/15.9 → processed(122B抽取) 16.3/26.8, 弃答23%
文献(严格EM, 7B系): RAG 6.2 / IRCoT 7.0 / Search-R1 19.6 / R3 paper 21.7 / ReSearch 22.3 / R1-Searcher 28.2
文件: eval_b_r3_musique_*/answers_short.jsonl; 矩阵档案 experiments/2026-08-12_baseline_matrix_musique/

## 6. Judge-C(诊断器 + 负结果)

组件验收(judge_v2, 双契约): MuSiQue AUC 0.9961 / 2Wiki零样本 0.9865 /
反事实塌陷 1.0000 / 闭卷泄漏 0.0000 / 分族假阳≤1.9% / ECE 0.0132
系统级(v3-RL轨迹): AUC 0.742, 分离 0.328/0.199; 失败分型 断链246/open70/全覆盖仍错4
选择性预测曲线: 答5%→79%, 10%→63%, 全解算点 5.8%@82-89%
负结果链: best-of-4 全策略(26.6-27.0) < majority 28.3 (oracle 35.0);
死因=计划-路径错位(排除法: 阈值/转述数据/形变均洗清; 槽1仅60%即直接证据)
文件: /scratch/boyuz5/acec/judge/ 下 j4v2_scores / phase1a_* / rerank_results2.json
教材: judge_train.jsonl 43.4万(六族+别名), spec=ACEC_JUDGE_SPEC.md

## 7. 训练动力学素材(§笔记=ACEC_V8_STOPPING_DESIGN_NOTES.md §1-9)

- 双档KL锚(0.02动作/0.5自由); 重锚失败实验(7集 ftk 0.7→7.2); 六连崩法医学
- 退化三重观测: 连续~ep183 / 冷重启~ep160 / 守护判据自动截停; slot句法退化样本在 diag_* 目录
- RL终点对SFT起点不敏感(47.5±1.6 vs 46.75); 规范plan使SFT 4hop 11→25
- 幻觉审计三代全过(v2.2/ep120/v3), 禁词命中≤0.07%

## 8. 复现信息

配置=run_configs/*.sh 逐参数; 训练器 rag/train/ppo_protocol_v8.py;
协议 rag/src/agent/protocol_v1.py + slot_schema.py; 语料构建 ~/acec/build_judge_data.py 等。
在跑: J8(观测注入消融), 结果~8/24 前, 由实验窗口负责, 出数后更新本文件 §9。
