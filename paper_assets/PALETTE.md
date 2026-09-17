# 论文配色方案(2026-09-16,用户给定七色)

| 色 | HEX | 相对亮度 | 角色 | 用在哪 |
|---|---|---|---|---|
| 天蓝 | `#8BC1E9` | 0.50 | **ours / 主方法**(query-level credit) | 柱、线、主行 |
| 沙金 | `#EDCD98` | 0.64 | 对照臂 A(trajectory-level coverage) | 柱、线 |
| 灰紫 | `#B0A3B8` | 0.39 | 对照臂 B / 基线(outcome-only,外部系统) | 柱、线 |
| 玫红 | `#e2989f` | 0.41 | 负向 / 病理(cost ordering、失效臂)、强调 | 少量 |
| 淡紫 | `#CAA6D2` | 0.44 | 第二强调(SFT 起点、参考线) | 少量 |
| 浅粉 | `#ecc6c5` | 0.62 | 强底纹(面板底、区间带) | 填充 |
| 米白 | `#faedea` | 0.87 | 弱底纹(表格主行 `\rowcolor`、图注底) | 填充 |

三臂灰度可分:0.50 / 0.64 / 0.39,黑白打印不混。

**文字用深色版(ink)**:浅色直接写字对比度不够,标注/数字用同色系加深 40%:
蓝 `#537490`、沙 `#8e7b5b`、灰紫 `#6a626e`、玫红 `#885b5f`、淡紫 `#796470`。

**代码**:`paper_assets/style.py` 的 `PAL`(fill)与 `INK`(文字),所有 `fig_*.py` 从这里取色;
LaTeX 端 `main.tex` 里 `\definecolor{shade}{HTML}{FAEDEA}` 等同名定义,表格主行 `\rowcolor{shade}`。

首用:`fig_main_arms.py`(§6.1 三臂图)。
