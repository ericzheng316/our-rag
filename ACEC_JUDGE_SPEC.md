# ACEC Judge-C 可执行规格(v1,2026-08-15 用户批准)

总纲:**双通道隔离**。奖励通道维持现状(gold 标题规则塑形,仅训练期);
judge 通道 = Coverage(s_i | C_t),**永不进奖励**,只用于作答门控、逐槽诊断、
Phase-2 观测注入消融。judge 失败不污染主线。

## 1. 协议 v1.2:规范 plan(蒸馏端解决槽本体)

- plan 每行 = 原子单跳问句 + 显式 #k 引用 + 期望答案类型:
  `2. Where was #1 born? (city)`
- teacher 重滚时由生成 prompt 强制;SFT v3 后策略自然产出。
- 推理期 schema parser 校验:2–6 行、每行可解析(问句+类型)、#k 引用无环
  且 k 指向已有行;不合法走现有 0.2 协议罚(不新增罚项)。
- 非抽取型答案(comparison/yes-no/聚合)在类型标注里显式声明,门控走旁路
  (只查路径一致性,不查值匹配)——2Wiki 迁移的 comparison 大族必需。

## 2. 运行期槽状态机

- 三态:`blocked`(含未解 #k)→ `open`(引用全绑定)→ `filled`(判定覆盖,
  记录 值/支持 span/段落 id)。
- 绑定值必须从 C_t 抽取且 span 回指到具体段落;参数记忆补的绑定一律无效。
- 候选:自适应 top-K——保留与 top-1 绑定分差 ≤δ 的候选 + 全局路径预算上限
  (拒绝死数字 2-3;δ 与预算由 gold 轨迹上 recall@K 平台标定)。
- 下游槽对每候选分别实例化;门控只看 DAG 上联合分最高的一条一致路径。
- 判定每 turn 对全量 C_t 重算(桥接证据后到 → 旧段落重判 → 状态级联翻转);
  (命题, 段落) 结果按字符串键缓存,每 turn 实际只算新增对
  (新段落×open 槽 + 新解锁槽×全部段落)。

## 3. Judge 本体:Qwen3.5-4B(用户定,无尺寸消融)

- 输入:绑定后原子问句 + 候选值 + **单个段落**(MuSiQue 官方 decomposition
  每步单段支撑,单段假设由数据构造保证)。
- 固定模板输出:支持判定 + 从段落逐字引用的支持 span。
- 分数 = "支持" token 概率,分桶温度校准;span 串匹配校验,段落中不存在
  该文字 → 强制判否(grounding 结构保证)。
- 槽级分数 = 跨段落 **max**(任何随段落数单调的聚合都可被词面堆积攻击)。
- 同家族红利:与策略 9B/teacher 122B 分词器逐字节一致 → span 边界/缓存键/
  alias 匹配无跨家族分词裂缝;vLLM/adapter/engine_pool 基建零改造。

## 4. 训练数据(gold 自动展开,不依赖 teacher 重滚 → 与 J5 并行)

- 正例:MuSiQue train decomposition step × gold 中间答案绑定 × gold 段落
  (5 万+,自带 span)。
- 负例六族:①官方闭池干扰段(构造性易混,免费最难负例);②gold 段实体替换;
  ③真段落×错误绑定;④同实体关系错配/反向;⑤hedge/否定改写;
  ⑥空上下文 + leave-one-out(被引段撤出 C_t 后判定必须塌陷)。
- 族⑥双职:训练目标 + 验收指标(反事实塌陷率≈100%、闭卷泄漏率≈0),把
  "以可见上下文为条件、可归因"变成可测试性质。
- 闭卷泄漏验收含**关系盲探针族**:段落同提两实体但不含目标关系(4B 先验
  最易补空处)。
- 跨数据集扩充:2Wiki `evidences` 三元组 + Hotpot supporting_facts 合成
  claim,防过拟合 MuSiQue 关系类型。

## 5. 校准与版本纪律

- 训练题带 gold → 每条 on-policy rollout 免费产出 (槽, C_t, 特权标签);
  滚动 buffer 按 闭池/开卷 × 根槽/依赖槽 分桶温度缩放;held-out on-policy
  ECE 超阈触发重校准。
- 观测注入期 judge 阶段内冻结,只在阶段边界换版。

## 6. 部署三级(每级独立可发表口径)

- **Phase 1a 离线诊断+重排(零干预)**:逐槽诊断(blocked 断链深度 vs open
  未查到,分开统计);coverage-vs-EM 条件相关给闭池晚期增益做机理归属
  (覆盖随 EM 涨=池结构利用;EM 涨覆盖不涨=参数猜测);best-of-N coverage
  加权重排。
- **Phase 1b 拒绝-重采样门控(controller 口径,单独标注)**:一致路径联合分
  ≥τ 且 answer alias-匹配路径末端 → 放行;拦截 → 回退重采样(**动作永远由
  策略生成**,环境不代写);[searches left: 0] 无条件放行(门控不得制造
  撞上限弃答)。
- **Phase 2 观测注入 RL 消融**:环境侧行 `[slots: 1✓(Nolan) 2… 3✗blocked]`
  (user-role token,不经 0.5 自由锚,稳定性结构零成本);风险="挑 judge
  爱看的文档",开卷侧反事实覆盖率监控兜底;与 B 控制臂同窗口(ep120 起
  ~38 集)对比。

## 7. 验收清单(J4 三门,不过不进 J5)

干扰段-only 假阳率 / 反事实塌陷率 / 闭卷泄漏率(含关系盲) / 绑定值与 gold
中间答案一致率 / 分桶 ECE 与 AUC(门槛 0.75) / coverage-at-answer × EM
条件相关 / 2Wiki+Hotpot 零样本 AUC。
系统级测床(on-policy 轨迹)在 SFT v3 之后用新轨迹执行(旧 236 轨迹 plan
非规范格式,只作参考不作门)。

## 8. 执行序列与资源

J1 schema/状态机/parser(CPU,现在)→ J2 数据六族(CPU)→ J3 4B LoRA 微调
(1 卡,shaping-off 后卡缝)→ J4 三门验收 → J5 teacher 全量重滚+SFT v3
(4 卡主槽 1-2 天;用户拍板不做 plan 回填)→ J6 Phase 1a → J7 Phase 1b →
J8 Phase 2。judge 推理布局:rollout 期借 DDP 卡余量(实测 42-52GB 空闲,
且 rollout 期 DDP 卡空转,时序错峰)。
与主线并行不冲突:shaping-off 验收、SFT 迁移基线、种子×2 按卡缝穿插。
