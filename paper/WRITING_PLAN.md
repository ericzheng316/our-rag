# 写作总计划(§1 → 投稿)

**题目**: Breaking Answer Ties: Evidence-Coverage Rewards for Deep Multi-Hop Search
**目标**: ARR 2026-10-15
**主叙事一句话**: 深多跳搜索里大多数 rollout 组答案全错,终局奖励排不出序,
剩下的成本项还教模型便宜地放弃;我们用 gold 证据覆盖给打平组一个有意义的排序,
换来种子一致的深链增益,并画出它的适用边界。
**已定决议**: 4hop 是因果主张的主端点(总量 EM 照报但不押注);122B 教师公开披露;
显式种子批进主表、同种子复跑批进附录。

---

## 一、逐节计划

| 节 | 内容 | 用什么证据 | 状态 |
|---|---|---|---|
| §1 Intro | 组级诊断 → 三派解法不够 → 覆盖修复 → 四发现 → 四贡献 | 占位数字 [84]% 等 | **已写 v2**,等数字回填 |
| §2 Related Work | 三小节:检索RL一系 / 过程奖励一系 / 打平组缓解四派(丢弃·合成·改结构·我们=给内容) | related_notes.md 的划界表 | 骨架已定,**最后写**;前置:精读 Search-P1 和 AVSPO、9 月新文献补扫 |
| §3 饥饿的形式化 | 轨迹/组定义、answer-tied 定义、覆盖优势分解公式、tie 账本(按 hop)、成本病理小面板(全错组 advantage×检索次数) | K=8 复算数字;noshape 训练日志(离线) | 框架**现在可写**,数字等 K=8 |
| §4 方法 | 奖励公式(首次命中+门控+成本项)、"轨迹排序不是步级信用"声明、双档 KL(降格为稳定性组件)、算法框 | 全部现成 | **立即可写** |
| §5 实验设置 | 模型/SFT(**含 122B 教师披露**)/数据/闭池vs开放口径/配对协议/统计口径(逐种子配对差,不装置信区间)/选点纪律(ep100 固定为主,ep120 沉附录) | HANDOFF §11 种子账目 | **立即可写** |
| §6 主结果 | Table 1 配对多种子(4hop 主端点)、Table 2 同底座阶梯、深度效应图、检索次数不升 | **等配对 noshape 种子** | 等数据,框架可先搭 |
| §7 机制 | 为什么有效:tie 账本 → 覆盖恢复排序的组占比(乘数列)、tie-aware AUC 修正后的诚实表述("覆盖不预测答对,它是进展监督") | **等 K=8 复算** | 等数据 |
| §8 迁移与边界 | 2Wiki 强 / Hotpot 小 / 开放 MuSiQue 零;reachability 分解 P(对)=P(够得着)×P(够得着后对) | **等带轨迹的 wiki18 重跑** | 等数据;表可先做 |
| §9 信号安置 | 训练奖励✓ / 离线诊断✓(选择性预测 5%→79%)/ 运行期控制✗ / 注入观测✗;措辞纪律:z=−1.80 写 "no reliable improvement" | J7/J8 材料全齐 | **可写**,控制在 0.75 页内 |
| §10 Limitations | 教师披露、闭池训练域、单训练集、标题匹配依赖文档级标注、KL 未单独消融、选点与种子不确定性分开报 | beta §10 清单 | 可写 |
| §11 Conclusion | 五句收束;"fix discrimination, not reachability" 再现一次 | — | 最后写 |
| Abstract | beta §4 骨架 | **三门全过才定稿**(见下) | 最后写 |

## 二、欠账清单(intro 里每个方括号对应一个在跑的实验)

| 占位数字 | 来源实验 | 状态与预计到账 | 到账后改哪里 |
|---|---|---|---|
| [+2.7, +7.2] 4hop 区间、"outcome-only counterpart" 措辞 | 配对 noshape 种子 1/2/3(同 seed、β=0) | nsd1 在跑,2/3 在续链,**~08-31** | §6 Table 1、intro 发现一升级为 paired 措辞;若配对总量差全显著→按可逆条款升回总量主张 |
| [84]%、[75/71/84]%、[×3.0]、[31–46]% | K=8 + v2.2 组分析复算(采样 15/24 通道已有,9 个在续链) | 采样 ~08-29 齐,复算离线 | §3 tie 账本、§7、intro 第 2 段与发现二;**K=8 下占比必然缩水,提前接受** |
| "fix discrimination not reachability" 从口号变测量 | noshape/run7 两臂 wiki18 重跑(带轨迹 msgs)→ reachability 分解 | 续链收官后用热基建补,**~09 月初** | §8 全节 + Fig 3 终稿形态 |

**决断日 9/25**:配对种子若不全,启用 scope-narrowing 措辞(已批),abstract 不做人质。

## 三、图表清单

| 编号 | 内容 | 依赖 | 生成器 |
|---|---|---|---|
| Fig 1 | 机制双面板:A=同为全错的 K 条轨迹,outcome-only 按成本排 vs +coverage 按证据进展排;B=打平组占比按 hop | K=8 复算 + noshape 日志 | 新写 fig_mechanism.py |
| Fig 2 | 逐种子 4hop ΔEM + ΔSearch | 配对种子 | 新写 fig_paired_seeds.py |
| Fig 3 | 迁移边界:x=证据可达性, y=配对 ΔEM(临时形态:四场景柱状) | reachability 分解 | 新写 fig_reachability.py |
| Fig 4(可选/附录) | 信号安置地图(训练✓/诊断✓/控制✗/观测✗) | 已齐 | fig_j8/fig_risk_coverage 改造 |
| Table 1 | 配对多种子主表 | 配对种子 | tab_main.py 改造 |
| Table 2 | 同底座阶梯(零样本→SFT→outcome-RL→+coverage) | 已齐 | tab_main.py |
| Table 3 | 开放/迁移边界(EM/F1/hop/检索/可达性) | reachability | tab_transfer.py 扩 |
| Table 4(次要) | 六方同台(caveat 全挂) | 已齐 | tab_baselines.py |
| 附录 | ftk 法医图、曲线全景、风险-覆盖曲线、六连崩表、K=8 细节 | 已齐 | paper_assets 现有 |

## 四、日程(今天 08-28 → 10-15)

- **08-28 → 08-31**:写 §4、§5(材料全齐);§3 框架;链上收官(nsd1-3 + K8 补齐)
- **09-01 → 09-07**:配对数字到账 → Table 1 + Fig 2;K=8 复算 → §3/§7 填数 + Fig 1;
  排 wiki18 带轨迹重跑 → reachability
- **09-08 → 09-14**:写 §6、§7、§8(含 Fig 3)、§9
- **09-15 → 09-21**:精读 Search-P1/AVSPO + 9 月新文献补扫 → 写 §2;写 §10、§11
- **09-22 → 09-28**:写 abstract;Overleaf ARR 模板组装全稿;图表统一终审
- **09-29 → 10-08**:内部评审一轮(≤3 agent)→ 修;引用第三遍核对(清 bib 里全部 [T])
- **10-09 → 10-14**:冻稿缓冲
- **10-15**:提交

## 五、冻稿前检查清单(精选)

- [ ] 配对 noshape 种子落地或按决断日走 scope-narrowing
- [ ] K=8 复算替换全部 [K8] 占位;AUC 用 tie-aware 实现
- [ ] reachability 分解完成,否则 §8 降格为描述、标题候选里删 reachability 字样
- [ ] 全文无 "zero gradient" / 无步级信用暗示 / 无 "pre-registered"(用 stopping criteria fixed in advance)
- [ ] ep100 固定检查点为主行,ep120 沉附录
- [ ] Table 4 图注挂全 caveat(底座/亲训/复现锚);StepSearch 补自报数脚注
- [ ] 教师披露 + 种子账目按 HANDOFF §10c/§11 口径
- [ ] 引用 [T] 标记清零;StepSearch/ReSearch 换正式 venue
- [ ] 图表全部由 paper_assets/paper 生成器可复现
