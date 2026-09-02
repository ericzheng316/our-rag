"""§3.3 成本病理面板: 全错组内 outcome-only 优势 vs 检索次数/覆盖 的相关.

数据: K=8 采样通道 (SFT v2.2 起点, 训练温度 0.7, n=799 题) —— 与 §3 tie 账本
同一测量协议. 训练日志只有 episode 级汇总, 逐 rollout 奖励在此离线重放.

奖励重放 = ppo_protocol_v8.py episodes_to_trajs 的逐字实现, 参数取实际训练
配置 (run_configs/ablate_noshape.sh): retrieval_cost=0.015, token_cost=1e-3,
format_penalty=0.2(默认), turn_limit_penalty=0.3, answer_weight=1.0,
max_turns=8, shaping_beta=0 (outcome-only 臂).

统计: 全错组 (8/8 alias_em=0) 内 LOO 优势 A_i = R_i - mean_{j!=i} R_j;
组内 Spearman(A, n_search) 与 Spearman(A, coverage), 均分 (平均秩处理并列),
汇报 mean per-group rho 及按 hop 分解.

跑法: source ~/acec/env.sh && $PYTHON paper_assets/analysis_cost_ordering.py
"""
import glob
import json
import math
import re
from pathlib import Path

D = "/home/boyuz5/acec/logs/k8_sampling_20260828T103201Z"
POOL = "/home/boyuz5/acec/musique/dev_answerable_pool.jsonl"
OUT = Path("/home/boyuz5/our-rag/paper_assets/out/analysis_cost_ordering.md")
NAME = "sftv22"

RETRIEVAL_COST = 0.015
TOKEN_COST = 1e-3
FORMAT_PENALTY = 0.2
TURN_LIMIT_PENALTY = 0.3
ANSWER_WEIGHT = 1.0
MAX_TURNS = 8

_RESULT = re.compile(r"<result>\n?(.*?)\n?</result>", re.DOTALL)
# 与训练器 free_strip 逐字一致 (think 标签及内容按自由 token 计价)
_FREE_STRIP = re.compile(
    r'<search slot="\d+">.*?</search>|<answer>.*?</answer>|<plan>.*?</plan>',
    re.DOTALL)

import sys
sys.path.insert(0, "/home/boyuz5/our-rag/rag/src")
from agent.protocol_v1 import parse_assistant_turn, AnswerAction, InvalidAction

from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def free_token_count(turn_text: str) -> int:
    free = _FREE_STRIP.sub("", turn_text).strip()
    if not free:
        return 0
    return len(TOKENIZER(free, add_special_tokens=False).input_ids)


def retrieved_titles(msgs):
    seen = set()
    for m in msgs:
        if m["role"] != "user":
            continue
        for block in _RESULT.findall(m["content"]):
            for para in block.split("\n\n"):
                if para.strip():
                    seen.add(para.strip().split("\n")[0].strip().strip('"').casefold())
    return seen


def replay_return(msgs, em: float):
    """逐 turn 重放 outcome-only 奖励, 返回 (R, n_search_turns, n_invalid, gave_up)."""
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    actions = [parse_assistant_turn(t) for t in assistant]
    answered = any(isinstance(a, AnswerAction) for a in actions)
    gave_up = (not answered) and len(assistant) >= MAX_TURNS
    R, n_search, n_invalid = 0.0, 0, 0
    for i, (text, act) in enumerate(zip(assistant, actions)):
        if isinstance(act, AnswerAction):
            R += ANSWER_WEIGHT * em
        elif isinstance(act, InvalidAction):
            R += -FORMAT_PENALTY
            n_invalid += 1
        else:  # SearchAction (含首 turn plan+search)
            R += -RETRIEVAL_COST
            n_search += 1
            if gave_up and i == len(assistant) - 1:
                R += -TURN_LIMIT_PENALTY
        R += -TOKEN_COST * free_token_count(text)
    return R, n_search, n_invalid, gave_up


def spearman(xs, ys):
    """平均秩 Spearman; 任一侧无方差返回 None."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    golds = {}
    for line in open(POOL):
        r = json.loads(line)
        golds[r["id"]] = set(t.casefold() for t in r["gold_titles"])

    ks = sorted(int(f.rsplit("_k", 1)[1].split(".")[0])
                for f in glob.glob(f"{D}/hops_{NAME}_k*.jsonl"))
    groups = {}
    for k in ks:
        msgs_by_id = {}
        for line in open(f"{D}/msgs_{NAME}_k{k}.jsonl"):
            r = json.loads(line)
            msgs_by_id[r["id"]] = r["messages"]
        for line in open(f"{D}/hops_{NAME}_k{k}.jsonl"):
            r = json.loads(line)
            msgs = msgs_by_id[r["id"]]
            em = float(r["alias_em"])
            R, n_search, n_invalid, gave_up = replay_return(msgs, em)
            gt = golds[r["id"]]
            cov = len(retrieved_titles(msgs) & gt) / len(gt)
            e = groups.setdefault(r["id"], {"hop": r["hop"], "rows": []})
            e["rows"].append({"em": em, "R": R, "ns": r["n_search"],
                              "ns_replay": n_search, "inv": n_invalid,
                              "gave_up": gave_up, "cov": cov})

    groups = {i: e for i, e in groups.items() if len(e["rows"]) == len(ks)}
    K = len(ks)

    # n_search 口径核对: hops 里的 n_search vs 重放解析出的 search turn 数
    mismatch = sum(1 for e in groups.values() for r in e["rows"]
                   if r["ns"] != r["ns_replay"])
    total_rows = sum(len(e["rows"]) for e in groups.values())

    lines = [f"# §3.3 cost-ordering panel — {NAME}, K={K}, groups={len(groups)}",
             "",
             f"reward replay: retrieval_cost={RETRIEVAL_COST}, token_cost={TOKEN_COST}, "
             f"format_penalty={FORMAT_PENALTY}, turn_limit_penalty={TURN_LIMIT_PENALTY}",
             f"n_search 口径核对: {mismatch}/{total_rows} rows 与 hops 记录不一致",
             ""]

    def block(tag, sel):
        af = {i: e for i, e in groups.items() if sel(e)
              and all(r["em"] == 0 for r in e["rows"])}
        rho_ns, rho_cov = [], []
        skip_ns = skip_cov = 0
        for e in af.values():
            Rs = [r["R"] for r in e["rows"]]
            mean = sum(Rs) / K
            A = [(K / (K - 1)) * (x - mean) for x in Rs]
            s = spearman(A, [r["ns"] for r in e["rows"]])
            if s is None:
                skip_ns += 1
            else:
                rho_ns.append(s)
            s2 = spearman(A, [r["cov"] for r in e["rows"]])
            if s2 is None:
                skip_cov += 1
            else:
                rho_cov.append(s2)
        n = len(af)
        m_ns = sum(rho_ns) / len(rho_ns) if rho_ns else float("nan")
        m_cov = sum(rho_cov) / len(rho_cov) if rho_cov else float("nan")
        neg_share = (sum(1 for x in rho_ns if x < 0) / len(rho_ns)) if rho_ns else float("nan")
        lines.append(
            f"| {tag} | {n} | {m_ns:+.3f} ({len(rho_ns)} def, {skip_ns} const) "
            f"| {neg_share:.0%} | {m_cov:+.3f} ({len(rho_cov)} def) |")
        return n, m_ns, m_cov

    lines.append("| slice | all-failure groups | mean within-group rho(A, n_search) "
                 "| share rho<0 | mean rho(A, coverage) |")
    lines.append("|---|---|---|---|---|")
    block("all hops", lambda e: True)
    for h in (2, 3, 4):
        block(f"{h}-hop", lambda e, h=h: e["hop"] == h)

    # 弃答/invalid 构成, 供正文措辞
    af_all = {i: e for i, e in groups.items()
              if all(r["em"] == 0 for r in e["rows"])}
    n_rows = sum(len(e["rows"]) for e in af_all.values())
    n_gu = sum(1 for e in af_all.values() for r in e["rows"] if r["gave_up"])
    n_inv = sum(1 for e in af_all.values() for r in e["rows"] if r["inv"] > 0)
    lines += ["",
              f"all-failure rollouts: {n_rows}; gave-up {n_gu} "
              f"({n_gu/n_rows:.1%}), with-invalid-turn {n_inv} ({n_inv/n_rows:.1%})"]

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
