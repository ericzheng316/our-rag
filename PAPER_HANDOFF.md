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
- +20集分层: 注入 30.8 vs 对照 35.6(OOD 适应税); 行为: 注入臂 inv 更快抬头
- ACEC 判决地图终稿: 奖励侧✓(塑形,z=3.98) / 推理期决策✗(J7) / 可学习观测✗(J8);
  belief 存活形态=诊断器+选择性预测曲线; gold-free judge-as-reward 留下篇。
工程数据: 注入使 rollout 由 ~2.5 分钟/集 增至 ~5 分钟(judge 波次批判定), 无新病理。
措辞纪律(评审意见采纳): z=−1.80 不显著, 成文用 "no reliable improvement,
trend negative", 不用 "underperforms"。
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

## 10c. 归因阶梯(wiki18 同台, musique full dev 2417, 2026-08-25)

同 §10 检索条件。回答"分数高是不是底座+SFT 的功劳":

| 台阶 | alias_em | 2/3/4hop | 格式错 | 增益及显著性 |
|---|---|---|---|---|
| Qwen3.5-9B 零样本+协议 | 19.53 | 27.2/13.8/6.7 | 440 | — |
| +SFT v2.2 | 24.78 | 33.2/19.3/8.9 | 7 | +5.3, z=+7.34 |
| +RL ep120 | 26.52 | 34.5/21.6/11.1 | ~3 | +1.7, z=+2.72 |

对照: 零样本 vs ReSearch(22.1) z=−3.35(显著更低); SFT-only vs ReSearch z=+3.40。
结论措辞: (1) 光靠底座不赢(零样本显著低于亲训对手); (2) 底座+SFT 是跨线主力,
且 SFT 首先买到协议可靠性(格式错 440→7 ≈ 18% 题免于协议死亡); (3) RL +1.7 显著
且集中在深链(3hop +2.3 / 4hop +2.2), 与闭池难度单调结论互证。
注脚(2026-08-28 更正, 作者确认): SFT v2.2/v3 轨迹由 **122B 教师模型**生成
(gen_teacher_traces.py, 10,203 rollouts → 净化保留; plan 结构锚定 MuSiQue 官方分解),
即存在外部教师蒸馏——本节旧稿"无外部教师蒸馏"为误记, 以本条为准。披露口径:
教师只进 SFT 冷启动; RL/评测/部署全程仅 9B 策略(诊断另用 4B judge)。
归因防御: (a) 塑形消融两臂同 SFT 起点, 教师不混杂因果主张; (b) RL 终点对
SFT 起点不敏感(v3-RL 46.75 vs 47.5±1.6), 配方不依赖教师质量;
(c) 冷启动蒸馏为该线常规操作(StepSearch 等亦用强教师), 但须在 Limitations 明示。
base-swap 探针(Qwen2.5-7B-Instruct + 同协议零样本)在跑, 出分补此节。
文件: logs/baseline_evals/{a3_zeroshot,sft_only}_wiki18_musique.jsonl

§10c 补: base-swap 探针(2026-08-25): Qwen2.5-7B-Instruct + 同协议零样本 wiki18 =
9.64 (2hop 15.65/3hop 4.34/4hop 0.99, 格式错 532≈22%, think 率 34.6%)。
同协议底座差: 19.53 vs 9.64 = +9.9(含协议适配折损, 非纯底座)。训练提升对比:
他们 +12.5 (9.6→22.1 ReSearch) vs 我们 +7.0 (19.5→26.5)。
论文口径: 跨系统表 = 环境行情; 贡献主张压同底座阶梯(SFT z=7.34 / RL z=2.72)。
文件: logs/baseline_evals/qwen25_7b_zeroshot_wiki18_musique.jsonl

## 10d. 塑形消融·开放语料全套(严格配对 ep100, wiki18 同台, 2026-08-26)

run7(β=0.3) vs noshape(β=0), 同训练量 ep100, 其余配方全同; ep120 列为参考(多训 20 集)。

| wiki18 开放 | 塑形 run7 | 无塑形 | 配对 z | ep120 参考 |
|---|---|---|---|---|
| MuSiQue 2417 | 25.69 | 26.40 | −1.41(无显著差) | 26.52 (vs 无塑形 z=+0.23) |
| — 4hop | 11.11 | 10.86 | +0.26(闭池 +7.2 在开放归零) | 11.11 |
| Hotpot 2k | 46.40 | 46.30 | +0.19 | 47.15 |
| — bridge | 41.13 | 42.19 | −1.78 | 41.19 |
| — comparison | **67.50** | 62.75 | **+3.80 显著** | 71.00 |
| 2Wiki 2k | **39.95** | 37.90 | **+3.01 显著** | 41.75 (z=+5.16) |
| — 2wiki 4hop | 33.03 | 28.51 | +2.39 显著 | 33.48 |

**预登记预测命中**: 在 run7-2Wiki 出分前, 基于"塑形的开放语料价值集中于证据完备性
构成瓶颈的题型(comparison/并行支)"这一机制解释, 预测 2Wiki(comparison 重)应出
正效应——实测 z=+3.01 命中。见本窗口对话记录时间线。

**论文措辞修订(塑形主张的范围界定)**:
(1) 闭池(训练域): 因果成立, 4hop +7.2 z=3.98, 难度单调——原主张不变;
(2) 开放语料: 总量效应在 MuSiQue/Hotpot 衰减至不显著; 显著存活于 Hotpot comparison
(z=+3.80)与 2Wiki 全量(z=+3.01)——即证据完备性为瓶颈的结构;
(3) RL 整体的开放转化与塑形无关地成立: noshape-RL vs SFT z=+2.49, ep120 vs SFT z=+2.72;
(4) 机制链闭合: 迁移矩阵 comparison +11.7 ↔ 塑形 comparison +4.75 ↔ 2Wiki 预测命中;
(5) 效率注脚: 塑形臂检索次数少 ~7%(musique 3.47 vs 3.75)成绩持平。
文件: logs/baseline_evals/{run7ep100,noshape,ours_ep120}_wiki18_{musique,hotpot2k,2wiki2k}.jsonl
工程注: 2wiki 深链 × wiki18 长段落顶破 8192 上下文 → eval_musique_hops 引擎改 16384(容量参数, 不影响语义)。

## 11. 种子账目决议(2026-08-28, 作者拍板)

两批"三种子"是不同的实验对象, **都保留, 各司其职**:

| 批次 | run | 全量 EM | 本质 |
|---|---|---|---|
| 旧批(同种子复跑) | run7 / 0816 / 0817 | 48.20/48.53/45.68 | --seed 全部为默认 20260805(数据序恒定), 差异=未播种采样+硬件非确定性 |
| 新批(显式种子) | sd1/sd2/sd3 (commit 7addc4b) | 47.12/47.83/[seed3待出] | seed 1/2/3 全链播种(数据序+torch+引擎采样) |

- **Table 1(主表)用新批**: 与 noshape 配对链(同 seed, β=0)构成 run 级严格配对
  ——同数据序、同引擎采样种子, 分歧只能来自奖励项。
- **旧批降级为附录**, 但有独立价值: 它测的是"固定课程下的纯训练随机性"
  (σ≈1.6), 与新批的"跨种子方差"构成两个正交的稳健性维度; 论文可报
  "fixed-seed rerun variance vs cross-seed variance", 这是多数同类论文不报的。
- 措辞: 旧批称 "same-seed independent reruns", 新批称 "explicit seeds 1/2/3";
  两者绝不混称、绝不合并统计。
