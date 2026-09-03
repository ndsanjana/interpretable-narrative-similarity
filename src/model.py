from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
import graph_builder as gb

CAUSAL_RELATION = ("action", "causes", "action")


@dataclass
class ModelConfig:
    text_dim: int = 768
    hidden: int = 256
    out_dim: int = 256
    layers: int = 3
    conv: str = "sage"
    heads: int = 4
    dropout: float = 0.1
    residual: bool = True
    readout: str = "multi"
    head_layers: int = 1
    pool_residual: bool = False
    pool_residual_gate_init: float = 0.0
    readout_concat_input: bool = False
    attn_uniform_init: bool = False
    node_text_dropout: float = 0.0
    plot_emb: int = 16
    status_emb: int = 8
    kind_emb: int = 8
    aggr: str = "sum"
    use_text: bool = True
    use_description: bool = True
    use_function: bool = True
    use_causal: bool = True
    use_reverse: bool = True
    use_story_scalar: bool = True
    use_story_interaction: bool = True
    use_plot_type: bool = True
    drop_relations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return asdict(self)


def selected_edge_types(cfg: ModelConfig) -> List[Tuple[str, str, str]]:
    edges = list(gb.EDGE_TYPES) if cfg.use_reverse else list(gb.FORWARD_RELATIONS)
    if not cfg.use_causal:
        rev = gb.reverse_relation(CAUSAL_RELATION)
        edges = [e for e in edges if e != CAUSAL_RELATION and e != rev]
    if cfg.drop_relations:
        drop = set(cfg.drop_relations)
        edges = [e for e in edges if e[1] not in drop]
    return edges


class InputEncoder(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.plot_emb = nn.Embedding(gb.N_PLOT_TYPES, cfg.plot_emb)
        self.status_emb = nn.Embedding(gb.N_NARRATIVE_STATUS, cfg.status_emb)
        self.kind_emb = nn.Embedding(gb.N_OUTCOME_KINDS, cfg.kind_emb)
        self.lin = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for nt in gb.NODE_TYPES:
            self.lin[nt] = nn.Linear(self.in_dim(nt), cfg.hidden)
            self.norm[nt] = nn.LayerNorm(cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def in_dim(self, nt: str) -> int:
        cfg = self.cfg
        if nt == "story":
            return cfg.plot_emb + cfg.status_emb + gb.STORY_FEATURE_DIM
        d = cfg.text_dim * (2 if nt == "action" else 1) + gb.SCALAR_DIMS[nt]
        if nt == "outcome":
            d += cfg.kind_emb
        return d

    def forward(self, data) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        out: Dict[str, torch.Tensor] = {}
        for nt in gb.NODE_TYPES:
            store = data[nt]
            n = store.num_nodes or 0
            scalar = store.x_scalar
            if nt == "story":
                dev = scalar.device
                plot = data["story"].plot_type.view(-1)
                status = data["story"].narrative_status.view(-1)
                pe = self.plot_emb(plot)
                if not cfg.use_plot_type:
                    pe = torch.zeros_like(pe)
                se = self.status_emb(status)
                sc = scalar
                if not cfg.use_story_scalar:
                    sc = torch.zeros_like(sc)
                elif not cfg.use_story_interaction:
                    sc = sc.clone()
                    sc[:, 21:] = 0.0
                feats = torch.cat([pe, se, sc], dim=-1)
            else:
                x = store.x
                if self.training and cfg.node_text_dropout > 0.0 and x.numel():
                    keep = (
                        torch.rand(x.shape[0], 1, device=x.device) >= cfg.node_text_dropout
                    ).float()
                    x = x * keep
                if nt == "action":
                    d = cfg.text_dim
                    desc, func = (x[:, :d], x[:, d : 2 * d])
                    if not cfg.use_description:
                        desc = torch.zeros_like(desc)
                    if not cfg.use_function:
                        func = torch.zeros_like(func)
                    x = torch.cat([desc, func], dim=-1)
                if not cfg.use_text:
                    x = torch.zeros_like(x)
                parts = [x, scalar]
                if nt == "outcome":
                    kind = store.kind.view(-1).long()
                    parts.append(self.kind_emb(kind))
                feats = torch.cat(parts, dim=-1)
            h = self.lin[nt](feats.float())
            h = self.drop(F.relu(self.norm[nt](h)))
            out[nt] = h
        return out


def _make_conv(cfg: ModelConfig, edge_types) -> HeteroConv:
    convs = {}
    for et in edge_types:
        if cfg.conv == "sage":
            convs[et] = SAGEConv((cfg.hidden, cfg.hidden), cfg.hidden)
        elif cfg.conv == "gatv2":
            assert cfg.hidden % cfg.heads == 0, "hidden must divide by heads"
            convs[et] = GATv2Conv(
                (cfg.hidden, cfg.hidden),
                cfg.hidden // cfg.heads,
                heads=cfg.heads,
                dropout=cfg.dropout,
                add_self_loops=False,
            )
        else:
            raise ValueError("unknown conv %r" % cfg.conv)
    return HeteroConv(convs, aggr=cfg.aggr)


class NarrativeGNN(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.edge_types = selected_edge_types(cfg)
        self.encoder = InputEncoder(cfg)
        self.convs = nn.ModuleList([_make_conv(cfg, self.edge_types) for _ in range(cfg.layers)])
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict({nt: nn.LayerNorm(cfg.hidden) for nt in gb.NODE_TYPES})
                for _ in range(cfg.layers)
            ]
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.multi = cfg.readout.startswith("multi")
        n_blocks = len(gb.NODE_TYPES) if self.multi else 1
        if cfg.readout == "multi_attn":
            self.queries = nn.ParameterDict(
                {
                    nt: nn.Parameter(
                        torch.zeros(cfg.hidden)
                        if cfg.attn_uniform_init
                        else torch.randn(cfg.hidden) / cfg.hidden**0.5
                    )
                    for nt in gb.NODE_TYPES
                    if nt != "story"
                }
            )
        if cfg.pool_residual and cfg.out_dim != cfg.text_dim:
            raise ValueError(
                "pool_residual needs out_dim == text_dim (%d vs %d)" % (cfg.out_dim, cfg.text_dim)
            )
        if cfg.pool_residual:
            self.pool_gate = nn.Parameter(torch.tensor(float(cfg.pool_residual_gate_init)))
        extra = cfg.text_dim if cfg.readout_concat_input else 0
        if cfg.head_layers > 1:
            self.head = nn.Sequential(
                nn.Linear(n_blocks * cfg.hidden + extra, cfg.hidden),
                nn.GELU(),
                nn.Linear(cfg.hidden, cfg.out_dim),
            )
        else:
            self.head = nn.Sequential(nn.Linear(n_blocks * cfg.hidden + extra, cfg.out_dim))

    def pooled_input_text(self, batch, nbatch: int) -> torch.Tensor:
        e = self.cfg.text_dim
        dev = batch["story"].x_scalar.device
        acc = torch.zeros(nbatch, e, device=dev)
        cnt = torch.zeros(nbatch, 1, device=dev)
        ones = None
        for nt in gb.TEXT_NODE_TYPES:
            store = batch[nt]
            x = getattr(store, "x", None)
            if x is None or x.numel() == 0:
                continue
            idx = store.batch
            x = x.float()
            chans = (x[:, :e], x[:, e : 2 * e]) if nt == "action" else (x,)
            ones = torch.ones(x.shape[0], 1, device=dev)
            for ch in chans:
                acc.index_add_(0, idx, ch)
                cnt.index_add_(0, idx, ones)
        return F.normalize(acc / cnt.clamp(min=1.0), p=2, dim=-1)

    def forward(self, batch) -> torch.Tensor:
        x_dict = self.encoder(batch)
        edge_index_dict = {
            et: batch[et].edge_index for et in self.edge_types if et in batch.edge_types
        }
        for layer, (conv, norms) in enumerate(zip(self.convs, self.norms)):
            out = conv(x_dict, edge_index_dict)
            new: Dict[str, torch.Tensor] = {}
            for nt, h in x_dict.items():
                o = out.get(nt)
                if o is None:
                    new[nt] = h
                    continue
                o = norms[nt](o)
                o = F.relu(o)
                o = self.drop(o)
                new[nt] = h + o if self.cfg.residual else o
            x_dict = new
        nbatch = int(batch["story"].batch.max().item()) + 1 if batch["story"].num_nodes else 0
        blocks = [global_mean_pool(x_dict["story"], batch["story"].batch, size=nbatch)]
        if self.multi:
            for nt in gb.NODE_TYPES:
                if nt == "story":
                    continue
                h = x_dict[nt]
                idx = batch[nt].batch
                if h.numel() == 0:
                    blocks.append(h.new_zeros((nbatch, self.cfg.hidden)))
                elif self.cfg.readout == "multi_sum":
                    blocks.append(global_add_pool(h, idx, size=nbatch))
                elif self.cfg.readout == "multi_attn":
                    score = (h * self.queries[nt]).sum(-1)
                    w = pyg_softmax(score, idx, num_nodes=nbatch).unsqueeze(-1)
                    blocks.append(global_add_pool(h * w, idx, size=nbatch))
                else:
                    blocks.append(global_mean_pool(h, idx, size=nbatch))
        pooled = None
        if self.cfg.readout_concat_input or self.cfg.pool_residual:
            pooled = self.pooled_input_text(batch, nbatch)
        if self.cfg.readout_concat_input:
            blocks.append(pooled)
        z = self.head(torch.cat(blocks, dim=-1))
        if self.cfg.pool_residual:
            z = pooled + self.pool_gate * z
        return F.normalize(z, p=2, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
