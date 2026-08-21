"""Render MuSiQue-Ans decompositions into protocol-v1 SFT trajectories.

Why this exists (2026-08-05): the SFT-v0 cold start (synthetic tasks, 62%
K=2) shows zero retrieval gain on MuSiQue 3/4-hop (EM(k) measurement,
~/acec/logs/hops_eval_*): the pipeline works on 2hop (+5pp) but the policy
never learned deep-chain bridging. MuSiQue train ships exactly the missing
supervision: gold sub-questions per hop, gold intermediate answers, and the
supporting paragraph for every step. Rendering those as ideal protocol-v1
trajectories teaches the two skills PPO needs to start from: (1) bind the
previous hop's discovered entity into the next query, (2) keep searching
past hop 2.

Rendering contract (mirrors synth/render_sft.py; protocol text imported,
never redefined):
  plan   — one line per decomposition step; ``#N`` back-references become
           "the answer from step N" (the plan is written before anything is
           discovered).
  search — slot k = step order (1-indexed); the query is the step's
           sub-question with ``#N`` resolved to the REAL previous answers —
           this is the bridge-binding supervision itself. ``#N`` is
           POSITIONAL (1-indexed step order), not step["id"] — keying by id
           silently resolves nothing (repo CLAUDE.md, 2026-08-04).
  result — the step's gold supporting paragraph plus sampled non-supporting
           paragraphs from the same question's 20-paragraph pool, shuffled
           (matches docs_per_search at inference; teaches selection under
           noise). Rendered "title. text", truncated at --max_doc_chars.
  answer — the official answer field, nothing else. No trailing
           confirmation search: the c_r analysis (same day) says never
           teach retrieval that buys no information.

Outputs: --out (sft jsonl: task_id, k, family, messages) and --tasks_out
(task_id, question, answer — the trainer's format_check input).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.protocol_v1 import SYSTEM_PROMPT  # noqa: E402

_REF = re.compile(r"#(\d+)")


def _norm_t(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def _resolve_refs(text: str, prev_answers: List[str]) -> str:
    """Replace positional #N with the actual answer of step N (1-indexed)."""
    def sub(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if 1 <= n <= len(prev_answers):
            return prev_answers[n - 1]
        return m.group(0)
    return _REF.sub(sub, text)


def _plan_refs(text: str) -> str:
    return _REF.sub(lambda m: f"the answer from step {m.group(1)}", text)


def _doc(p: Dict[str, Any], max_chars: int) -> str:
    body = str(p.get("paragraph_text", ""))[:max_chars]
    return f"{p.get('title', '')}. {body}"


def degrade_query(query: str, prev_answers: List[str]) -> str | None:
    """按实测主导失败型（unbound）退化：把已解析进查询的桥接实体抹回占位。
    没有可抹的实体时退化为"只留关系短语"（截掉主语）。返回 None = 无法退化。"""
    out = query
    hit = False
    for i, a in enumerate(prev_answers):
        if a and a in out:
            out = out.replace(a, f"the answer from step {i + 1}")
            hit = True
    if hit:
        return out
    parts = query.split(">>")
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()  # 只留关系短语（vague 型）
    return None


# ── v1.2 规范 plan(--canonical_plan)────────────────────────────────
_REL_T = {
    "location": "Where is {e} located?", "country": "In which country is {e}?",
    "place of birth": "Where was {e} born?", "date of birth": "When was {e} born?",
    "place of death": "Where did {e} die?", "date of death": "When did {e} die?",
    "spouse": "Who is the spouse of {e}?", "father": "Who is the father of {e}?",
    "mother": "Who is the mother of {e}?", "performer": "Who performed {e}?",
    "author": "Who is the author of {e}?", "director": "Who directed {e}?",
    "employer": "Who is the employer of {e}?", "genre": "What genre is {e}?",
    "record label": "What record label is {e} on?",
    "capital": "What is the capital of {e}?",
    "headquarters location": "Where is the headquarters of {e}?",
    "educated at": "Where was {e} educated?",
    "child": "Who is the child of {e}?", "sibling": "Who is the sibling of {e}?",
}


def _canon_q(q: str) -> str:
    q = q.strip()
    if ">>" not in q:
        return q if q.endswith("?") else q + "?"
    left, _, rel = q.partition(">>")
    e, rel = left.strip(), rel.strip().lower()
    t = _REL_T.get(rel)
    return t.format(e=e) if t else f"What is the {rel} of {e}?"


_QSTOP = {"who", "what", "where", "when", "which", "whom", "whose", "how",
          "is", "are", "was", "were", "did", "does", "do", "the", "a", "an",
          "of", "in", "on", "at", "to", "for", "by", "with", "and", "or",
          "that", "this", "it", "its", "as", "from", "into", "named", "name"}


def _keywordize(q: str) -> str:
    """canonical 模式的查询风格:剥问询词/停用词 → 关键词短语。
    与 plan 行文本构造性分叉(防查询=plan 拷贝的循环吸引子,v3 事故)。"""
    q = q.replace(">>", " ").rstrip("?")
    words = [w for w in q.split() if w.strip() and
             w.strip(",.'\"").lower() not in _QSTOP]
    return " ".join(words) if words else q


def _infer_type(ans: str) -> str:
    import re as _re
    a = ans.strip()
    if _re.fullmatch(r"1\d{3}|20\d{2}", a):
        return "year"
    if _re.search(r"\d{1,2},? \d{4}|January|February|March|April|May|June|July"
                  r"|August|September|October|November|December", a):
        return "date"
    if _re.fullmatch(r"[\d,.]+", a):
        return "number"
    if a.lower() in ("yes", "no"):
        return "yes/no"
    return "entity"


def render_record(rec: Dict[str, Any], rng: random.Random,
                  docs_per_result: int, max_doc_chars: int,
                  inject_rewrite: float = 0.0,
                  canonical_plan: bool = False) -> Dict[str, Any] | None:
    steps = rec.get("question_decomposition") or []
    paras = rec.get("paragraphs") or []
    answer = str(rec.get("answer", "")).strip()
    if not steps or not answer:
        return None
    by_idx = {p["idx"]: p for p in paras}
    non_support = [p for p in paras if not p.get("is_supporting")]

    if canonical_plan:
        plan_lines = [
            f"{i + 1}. {_canon_q(str(s['question']))} "
            f"({_infer_type(str(s.get('answer', '')))})"
            for i, s in enumerate(steps)]
    else:
        plan_lines = [f"{i + 1}. {_plan_refs(str(s['question']).strip())}"
                      for i, s in enumerate(steps)]
    plan = "<plan>\n" + "\n".join(plan_lines) + "\n</plan>"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(rec["question"]).strip()},
    ]

    prev_answers: List[str] = []
    skip_turns: List[int] = []
    asst_i = -1
    # 选一个可注入的跳（有前序实体的跳优先 = unbound 型可构造）
    inject_at = -1
    if inject_rewrite > 0 and rng.random() < inject_rewrite:
        candidates = [i for i in range(1, len(steps))]
        rng.shuffle(candidates)
        inject_at = candidates[0] if candidates else -1
    for i, s in enumerate(steps):
        gold = by_idx.get(s.get("paragraph_support_idx"))
        if gold is None:
            return None  # 支撑段落缺失 —— 轨迹不完整，整题丢弃
        query = _resolve_refs(str(s["question"]).strip(), prev_answers)
        if canonical_plan:
            query = _keywordize(query)
        if i == inject_at:
            bad_q = degrade_query(query, prev_answers)
            if bad_q is not None and bad_q != query:
                # 确定性闸门：坏查询在本题闭池检索下确实 miss 全部 gold 段
                from agent.episode import _title_of, pool_retriever
                retr = pool_retriever({"pool": [
                    {"id": f"p{p_['idx']}",
                     "contents": f"\"{p_['title']}\"\n{p_['paragraph_text']}"}
                    for p_ in paras]})
                bad_docs = retr(bad_q, docs_per_result)
                gold_titles = { _norm_t(p_["title"]) for p_ in paras
                               if p_.get("is_supporting") }
                if not ({_title_of(d["contents"]) for d in bad_docs} & gold_titles):
                    action_bad = f'<search slot="{i + 1}">{bad_q}</search>'
                    messages.append({"role": "assistant",
                                     "content": f"{plan}\n{action_bad}" if i == 0
                                     else action_bad})
                    asst_i += 1
                    skip_turns.append(asst_i)
                    block = "\n---\n".join(
                        _doc({"title": d["contents"].split("\n")[0].strip('"'),
                              "paragraph_text": d["contents"].split("\n", 1)[-1]},
                             max_doc_chars) for d in bad_docs)
                    messages.append({"role": "user",
                                     "content": f"<result>\n{block}\n</result>"})
        action = f'<search slot="{i + 1}">{query}</search>'
        messages.append({"role": "assistant",
                         "content": f"{plan}\n{action}" if (i == 0 and asst_i < 0)
                         else action})
        asst_i += 1

        distractors = rng.sample(non_support,
                                 min(docs_per_result - 1, len(non_support)))
        docs = [gold] + distractors
        rng.shuffle(docs)
        block = "\n---\n".join(_doc(p, max_doc_chars) for p in docs)
        messages.append({"role": "user", "content": f"<result>\n{block}\n</result>"})
        prev_answers.append(str(s.get("answer", "")).strip())

    messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    fam = str(rec.get("id", "?")).split("__")[0]
    return {"task_id": str(rec["id"]), "k": len(steps), "family": fam,
            "messages": messages, "skip_turns": skip_turns}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="musique_ans jsonl (official format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks_out", required=True)
    ap.add_argument("--max_2hop", type=int, default=4000,
                    help="2hop 截断上限（深链全收 —— 深度探索是目标）")
    ap.add_argument("--docs_per_result", type=int, default=3)
    ap.add_argument("--max_doc_chars", type=int, default=800)
    ap.add_argument("--inject_rewrite", type=float, default=0.0,
                    help="按此比例的题插入改写示范：程序退化查询（去桥接实体等，"
                         "按引导实验实测 unbound 是主导失败型）→ 闭池检索器真实"
                         "坏结果 → 同 slot 的 gold 子问题作为改写。坏查询 turn 记入"
                         " skip_turns（context-only）。仅在退化查询确实检索不到"
                         " gold（确定性判定）时插入。")
    ap.add_argument("--canonical_plan", type=int, default=0,
                    help="1 = v1.2 规范 plan(原子问句保 #k + 答案类型)")
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_family: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    with open(args.data, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("answerable", True):
                continue
            by_family[str(rec.get("id", "?")).split("__")[0]].append(rec)

    picked: List[Dict[str, Any]] = []
    for fam in sorted(by_family):
        pool = by_family[fam]
        rng.shuffle(pool)
        if fam == "2hop":
            pool = pool[: args.max_2hop]
        picked.extend(pool)

    n_ok = n_drop = 0
    fam_count: collections.Counter = collections.Counter()
    leak = 0
    with open(args.out, "w", encoding="utf-8") as out_fh, \
            open(args.tasks_out, "w", encoding="utf-8") as tasks_fh:
        for rec in picked:
            traj = render_record(rec, rng, args.docs_per_result,
                                  args.max_doc_chars, args.inject_rewrite,
                                  canonical_plan=bool(args.canonical_plan))
            if traj is None:
                n_drop += 1
                continue
            # 泄漏审计：最终答案不应出现在任何 search query 里
            # （中间答案出现在后续 query 是设计 —— 桥接绑定 —— 不算泄漏）
            ans_low = str(rec["answer"]).strip().lower()
            for m in traj["messages"]:
                if m["role"] == "assistant" and "<search" in m["content"]:
                    q = m["content"].split(">", 2)[-1].lower()
                    if ans_low and ans_low in q:
                        leak += 1
                        break
            out_fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
            tasks_fh.write(json.dumps({"task_id": traj["task_id"],
                                       "question": str(rec["question"]).strip(),
                                       "answer": str(rec["answer"]).strip()},
                                      ensure_ascii=False) + "\n")
            n_ok += 1
            fam_count[traj["family"]] += 1

    print(f"[render-musique] 轨迹 {n_ok}  丢弃 {n_drop}  "
          f"query含最终答案 {leak} ({leak / max(n_ok, 1):.1%})", flush=True)
    print(f"[render-musique] 家族分布 {dict(sorted(fam_count.items()))}", flush=True)


if __name__ == "__main__":
    main()
