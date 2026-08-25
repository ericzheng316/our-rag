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

## 9. J8 观测注入消融(终局,2026-08-23)

设计: v3-RL ep100 起双臂同窗口(INJ 注入 [slots:...] 环境行 vs CTRL 无), 60 集,
守护对称截停(INJ ep49 / CTRL ep50——退化臂无关,第五次观测)。
step_40 全量对比: INJ 46.71 vs CTRL 47.83, McNemar z=−1.80(99:126)→ 无显著效应。
动力学: 注入初期扰动(+20 集 −4.8, 分层)后适应至持平——策略消化而非利用信号。
归因: 信号保守有偏(计划-路径错位致填充率 60%/19%)+ 预算行已提供停机信息, 边际信息小。
预注册止损执行: 信念-作-观测关闭。信念线终账: 选择✗/门禁✗/注入✗/诊断✓/gold塑形✓(z=3.98)。
工程数据: 注入使 rollout 由 ~2.5 分钟/集 增至 ~5 分钟(judge 波次批判定), 无新病理。
文件: logs/j8_inj_20260821T235038Z / j8_ctrl_20260822T163135Z / eval_j8_ladder_*

## 9. J8 观测注入消融(终局,2026-08-23)

从 v3-RL 终点双臂同窗口(注入 [slots:...] 环境行 vs 对照),各 ~50 集(守护截停):
- +40集全量 dev: 注入 46.71 vs 对照 47.83 (−1.1pp, McNemar z=−1.80, 99:126)
- +20集分层: 注入 30.8 vs 对照 35.6(OOD 适应税); 行为: 注入臂 inv 更快抬头, rollout 慢~3×
- 判决: 信念-作-观测关闭(预注册判据)。归因脚注: 信号保守有偏(填充率60%/19%,
  计划-路径错位)+ 窗口在饱和平台上。
- ACEC 判决地图终稿: 奖励侧✓(塑形,z=3.98) / 推理期决策✗(J7) / 可学习观测✗(J8);
  belief 存活形态=诊断器+选择性预测曲线; gold-free judge-as-reward 留下篇。
文件: logs/j8_inj_20260821T235038Z / j8_ctrl_20260822T163135Z / eval_j8_ladder_20260823T065004Z

## 10. 六方 wiki18 开放检索对比(MuSiQue full dev 2417, 2026-08-23)

统一条件: 同一 wiki18 语料(21,015,324 段, 2018 dump)+ e5-base-v2 fp32 Flat 检索服务;
temp 0; 各系统用其官方协议逐字复现(来源: Search-R1 infer.py / ZeroSearch inference.py(与
Search-R1 逐字同款)/ ReCall 仓库 c820a43 历史版 ReSearch 模板与 pipeline / R1-Searcher
eval_search_loacl.py prompt v0)。度量: alias_em(SQuAD 归一, 含官方别名)。

| 系统 | 底座 | alias_em | F1 | 2hop | 3hop | 4hop | 均搜 |
|---|---|---|---|---|---|---|---|
| **ACEC ep120 (ours)** | Qwen3.5-9B+LoRA | **26.52** | 36.90 | 34.50 | 21.58 | 11.11 | 3.40 |
| ReSearch | Qwen2.5-7B-Inst | 22.09 | 32.43 | — | — | — | 3.26 |
| Search-R1 (ppo) | Qwen2.5-7B | 20.40 | 29.50 | — | — | — | 3.39 |
| StepSearch | Qwen2.5-7B-Inst | 18.25 | 26.96 | — | — | — | 2.11 |
| R1-Searcher | Qwen2.5-7B | 15.27 | 22.88 | — | — | — | 2.51 |
| ZeroSearch | Qwen2.5-7B | 9.43 | 16.78 | 13.66/5.00/4.69 | | | 1.21 |

配对显著性(同 2417 题): ours vs ReSearch 243:136, McNemar z=5.50;
ours vs Search-R1 288:140, z=7.15。
必注 caveat: (1) 底座不同(9B Qwen3.5 vs 7B Qwen2.5)——因果主张压在同底座
A 阶梯/R3 同栈/塑形消融上, 本表是环境行情对比; (2) ReSearch 在 MuSiQue train 上
亲训(最强 in-domain 对手, 仍 -4.4pp); (3) ZeroSearch 训练面对模拟 Google 文档,
真实 e5 检索是其分布外(均搜 1.21 即证), 分数须带此注; (4) 复现忠实度锚:
Search-R1 论文自报 MuSiQue ~19.6 vs 我们复现 20.4, ReSearch ~22 vs 22.1。
文件: logs/baseline_evals/{research,searchr1,stepsearch,r1searcher,zerosearch}_musique_dev.jsonl*
+ ours_ep120_wiki18_musique.jsonl; 驱动 ~/acec/baseline_infer.py; 检索服务 ~/acec/wiki18_server.py。
hotpot 2000 子集(bridge 1600/comparison 400)六方在跑, 出分后补 §10b。

## 10b. 六方 wiki18 HotpotQA 对比(统一 2000 题子集: bridge 1600 / comparison 400, 2026-08-23)

同 §10 条件; 子集 seed 20260823, 文件 baselines/eval_data/hotpot_dev2k.jsonl。
关键背景: Search-R1 在 nq+hotpotqa 上亲训(in-domain), R1-Searcher 训练数据亦含 hotpot;
我们对 hotpot 纯零样本(仅 MuSiQue 训练)。

| 系统 | alias_em | bridge | comparison | F1 | 均搜 |
|---|---|---|---|---|---|
| **ACEC ep120 (ours, 零样本)** | **47.15** | **41.19** | **71.00** | 59.76 | 2.69 |
| Search-R1 (in-domain) | 45.00 | 38.69 | 70.25 | 56.24 | 2.79 |
| ReSearch | 44.05 | 38.56 | 66.00 | 56.62 | 2.79 |
| R1-Searcher (含 hotpot 训练) | 39.60 | 34.19 | 61.25 | 51.07 | 2.01 |
| StepSearch | 35.15 | 28.94 | 60.00 | 44.86 | 1.78 |
| ZeroSearch | 30.70 | 23.62 | 59.00 | 41.45 | 1.08 |

配对(ours vs Search-R1, 同 2000 题): all 190:147 z=2.34(显著);
bridge 159:119 z=2.40(显著); comparison 31:28 z=0.39(打平其主场强项)。
叙事: 零样本迁移整体超过亲训 in-domain 系统; 复现锚: Search-R1 论文自报 hotpot ~43。
ZeroSearch 在 hotpot 塌得少(30.7 vs musique 9.4)与其"搜一次即答"习惯自洽(2 跳够用)。
文件: logs/baseline_evals/*_hotpot2k.jsonl* + ours_ep120_wiki18_hotpot2k.jsonl。
