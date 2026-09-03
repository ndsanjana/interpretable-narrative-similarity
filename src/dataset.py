from __future__ import annotations
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData
import graph_builder as gb


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class GraphCorpus:

    def __init__(self, data_dir: str, device: str = "cpu") -> None:
        graphs = torch.load(os.path.join(data_dir, "graphs.pt"), weights_only=False)
        npz = np.load(os.path.join(data_dir, "node_text_emb.npz"), allow_pickle=True)
        tale_ids = [str(t) for t in npz["tale_ids"]]
        self.text_dim = int(npz["dim"][0])
        self.encoder_name = str(npz["encoder"][0])
        blocks = {}
        for name in list(gb.TEXT_NODE_TYPES) + ["action_fn"]:
            blocks[name] = (npz["emb_" + name], npz["off_" + name])
        self.graphs: Dict[str, HeteroData] = {}
        for i, tale_id in enumerate(tale_ids):
            data = graphs[tale_id]
            for nt in gb.TEXT_NODE_TYPES:
                emb, off = blocks[nt]
                x = torch.from_numpy(np.asarray(emb[off[i] : off[i + 1]], dtype=np.float32))
                if nt == "action":
                    aemb, aoff = blocks["action_fn"]
                    xf = torch.from_numpy(np.asarray(aemb[aoff[i] : aoff[i + 1]], dtype=np.float32))
                    assert xf.shape[0] == x.shape[0], "action channels out of step for %s" % tale_id
                    x = torch.cat([x, xf], dim=-1)
                assert x.shape[0] == (
                    data[nt].num_nodes or 0
                ), "text rows do not match node count for %s/%s" % (tale_id, nt)
                data[nt].x = x
            self.graphs[tale_id] = data
        self.tale_ids = tale_ids
        self.index = {t: i for i, t in enumerate(tale_ids)}

    def __contains__(self, tale_id: str) -> bool:
        return tale_id in self.graphs

    def get(self, tale_id: str) -> HeteroData:
        return self.graphs[tale_id]

    def batch(self, tale_ids: Sequence[str]) -> Batch:
        return Batch.from_data_list([self.graphs[t] for t in tale_ids])


class TripletSet(Dataset):

    def __init__(self, corpus: GraphCorpus, rows: List[dict], hard_resample: bool = False) -> None:
        self.corpus = corpus
        self.rows = [
            r
            for r in rows
            if r["anchor_id"] in corpus and r["pos_id"] in corpus and (r["neg_id"] in corpus)
        ]
        self.dropped = len(rows) - len(self.rows)
        self.hard_resample = hard_resample
        self.cands: Dict[str, List[str]] = defaultdict(list)
        self.cand_type: Dict[str, str] = {}
        for r in self.rows:
            self.cands[r["anchor_id"]].append(r["neg_id"])
            self.cand_type[r["neg_id"]] = r["neg_atu"]
        for k in self.cands:
            self.cands[k] = sorted(set(self.cands[k]))
        self.current = [r["neg_id"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[str, str, str, str]:
        r = self.rows[i]
        return (r["anchor_id"], r["pos_id"], self.current[i], r.get("tier", "all"))

    def reset_negatives(self) -> None:
        self.current = [r["neg_id"] for r in self.rows]

    @torch.no_grad()
    def resample(self, model, device: str, batch_size: int = 64) -> int:
        if not self.hard_resample:
            return 0
        model.eval()
        ids = self.corpus.tale_ids
        embs = []
        for i in range(0, len(ids), batch_size):
            batch = self.corpus.batch(ids[i : i + batch_size]).to(device)
            embs.append(model(batch).detach().cpu())
        emb = torch.cat(embs, dim=0)
        idx = self.corpus.index
        changed = 0
        for i, r in enumerate(self.rows):
            cands = [
                c
                for c in self.cands[r["anchor_id"]]
                if self.cand_type.get(c) != r["pos_atu"]
                and c != r["pos_id"]
                and (c != r["anchor_id"])
            ]
            if len(cands) < 2:
                continue
            a = emb[idx[r["anchor_id"]]]
            sims = emb[[idx[c] for c in cands]] @ a
            pick = cands[int(torch.argmax(sims).item())]
            if pick != self.current[i]:
                changed += 1
            self.current[i] = pick
        model.train()
        return changed


def collate(items):
    a, p, n, tier = zip(*items)
    return (list(a), list(p), list(n), list(tier))


def make_loader(
    triplets: TripletSet,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    return DataLoader(
        triplets,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
