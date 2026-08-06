"""DDP 版 ppo_update 等价性：2 进程 gloo 分片的更新后权重必须与单进程一致。

用 test_ppo_v8 的 fake CausalLM（确定性、逐位置独立），同一批 trajs：
  单进程 ppo_update → 权重快照 A
  2 rank（gloo, CPU）ppo_update（rank0 给 trajs，rank1 给 None）→ 快照 B
断言 A == B（fp32 容差 1e-6）。守的是：broadcast、交错分片、梯度 SUM
all-reduce、每 rank 各自 step 的整条 DDP 数学链。
"""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.multiprocessing as mp

from ppo_rsf_vllm_v8 import ppo_update
from tests.test_ppo_v8 import _FakeCausalLM


def _make_setup(seed=11):
    torch.manual_seed(seed)
    model = _FakeCausalLM()
    for p in model.parameters():
        p.requires_grad_(True)
    vhead_lin = torch.nn.Linear(16, 1)

    class VH(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = copy.deepcopy(vhead_lin)

        def forward(self, x):
            return self.lin(x).squeeze(-1)

    vhead = VH()
    opt = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": 1e-3},
         {"params": vhead.parameters(), "lr": 1e-3}])
    return model, vhead, opt


def _make_trajs(seed=5):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for lens in [((6, 3), (8, 2)), ((5, 4),), ((7, 2), (6, 3), (9, 1))]:
        traj = []
        for li, lo in lens:
            inp = torch.randint(0, 64, (li,), generator=g)
            out = torch.randint(0, 64, (lo,), generator=g)
            r = float(torch.rand((), generator=g)) - 0.3
            traj.append((inp, out, r, None, 0.7))
        trajs.append(traj)
    return [[t] for t in trajs]


KW = dict(gamma=0.99, gae_lambda=0.95, clip_eps=0.2, value_clip=0.2,
          value_coef=0.5, kl_coef=0.0, entropy_coef=0.0, ppo_epochs=1,
          max_grad_norm=1.0, max_engine_logprob_mae=None,
          micro_turns=1, scan_batch=4)


def _run_single():
    model, vhead, opt = _make_setup()
    stats = ppo_update(model, vhead, opt, _make_trajs(), **KW)
    return ({k: v.detach().clone() for k, v in model.state_dict().items()},
            {k: v.detach().clone() for k, v in vhead.state_dict().items()},
            stats)


def _ddp_worker(rank, world, port, out_q):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model, vhead, opt = _make_setup()
    trajs = _make_trajs() if rank == 0 else None
    stats = ppo_update(model, vhead, opt, trajs,
                       dist_group=dist.group.WORLD, **KW)
    if rank == 0:
        out_q.put(({k: v.detach().clone() for k, v in model.state_dict().items()},
                   {k: v.detach().clone() for k, v in vhead.state_dict().items()},
                   stats))
    dist.barrier()
    dist.destroy_process_group()


class DDPEquivalenceTest(unittest.TestCase):
    def test_two_rank_matches_single(self):
        m_a, v_a, st_a = _run_single()

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [ctx.Process(target=_ddp_worker, args=(r, 2, 29517, q))
                 for r in range(2)]
        for p in procs:
            p.start()
        m_b, v_b, st_b = q.get(timeout=180)
        for p in procs:
            p.join(timeout=60)
            self.assertEqual(p.exitcode, 0)

        for k in m_a:
            self.assertTrue(torch.allclose(m_a[k], m_b[k], atol=1e-6),
                            f"model 权重不一致: {k}")
        for k in v_a:
            self.assertTrue(torch.allclose(v_a[k], v_b[k], atol=1e-6),
                            f"vhead 权重不一致: {k}")
        for k in ("policy_loss", "value_loss", "n_turns"):
            self.assertAlmostEqual(float(st_a[k]), float(st_b[k]), places=4,
                                   msg=f"stats[{k}] 不一致")


if __name__ == "__main__":
    unittest.main()
