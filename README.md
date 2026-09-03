# Interpretable Narrative Similarity

Code for a narrative-similarity system that decides which of two candidate folktales is closer to an anchor tale. Each tale is turned into a structured record by an LLM extractor, the record becomes a heterogeneous graph, a graph neural network embeds the graph, and the graph cosine is fused at score level with a text-embedding cosine. Because the fusion is additive with no post-fusion normalisation, every fused score decomposes exactly into a graph term and a text term, which is what makes the system interpretable.

Headline result on the ATU-Union benchmark (1,418 triplets, mean of 10 seeds):

| scorer | accuracy |
|---|---|
| graph-only | 78.84 (sd 1.79) |
| fused, learned scalar, Gemini text | 83.57 (sd 1.25) |
| fused, additive, EmbeddingGemma text (no API) | 82.70 (sd 1.42) |
| token Jaccard baseline | 67.91 |

Learned scalar weights: alpha 0.4344 (graph), beta 0.5656 (text). Decomposition identity R^2 = 1.0 for the additive and scalar arms. Channel angle 48.83 degrees in decision space and 59.85 degrees (mean principal angle) in representation space. The full table is in `results/score_table.md`, the raw aggregates in `results/score_aggregate.json`, and the interpretability report in `results/interpret_report.md`.

## Layout

```
src/
  schemas.py            extraction record schema and parse-and-repair
  extract.py            stage 1: tale text -> structured record via an LLM
  causal_pass.py        stage 2: causal-only second pass over the extracted action list
  merge_causal.py       stage 2b: write the causal parent sets back into the records
  eval_causal_pass.py   scores a causal pass against the 22 hand-annotated tales in data/
  graph_builder.py      record -> heterogeneous graph
  build_graph_data.py   graphs + EmbeddingGemma node-text encodings for the whole corpus
  encode_tales.py       full-tale EmbeddingGemma vectors (no-API text channel)
  gemini_embed.py       full-tale Gemini vectors (API text channel), hash-keyed cache
  split_union.py        type-disjoint train / validation split
  dataset.py, model.py  triplet dataset and the graph model
  train.py              trains one seed of the fixed configuration
  fusion.py             additive, learned-scalar and gated fusion arms; decomposition
  score.py              canonical scorer: graph, text, fusion arms, per seed, mean / sd / best seed
  interpret.py          who-decides table, channel angles, edge knockouts
prompts/                the two extraction prompts, parsed by the runners at start
configs/final.txt       the exact training flags and the 10 seeds
results/                outputs of the frozen final run (the numbers above)
data/                   22-tale causal ground-truth annotations
```

## Setup

Python 3.11. Install PyTorch for your platform first, then:

```
pip install -r requirements.txt
```

Extraction sends each tale to an OpenAI-compatible chat completions endpoint (`--url`, `--model`; the runs used `openai/gpt-oss-20b`). Node-text and full-tale encoding use `google/embeddinggemma-300m` (gated on Hugging Face; put `HF_TOKEN=...` in `.env`). The headline text channel uses `gemini-embedding-001` (2048-d) and needs `GEMINI_API_KEY=...` in `.env`; the EmbeddingGemma channel is the no-API alternative and is within one point of it after fusion.

## Data

The corpus is the Annotated Folktales collection from the trilogy project:

Hagedorn, J. (2022). trilogy: Annotated Folktales, release v1.1. Zenodo. https://doi.org/10.5281/zenodo.6575263 (CC-BY-SA 4.0). Repository: https://github.com/j-hagedorn/trilogy

No tale text or triplet file is distributed here. The benchmark was built from `aft.csv` as follows: keep tales of at least 100 words, drop ATU types with fewer than two usable tales, deduplicate within type at MiniLM cosine above 0.95, split 80/20 by tale type (seed 922): 137 training types and 33 held-out evaluation types, 1,418 tales in total. Positives are same-type pairs; negatives are drawn from other types, and letter variants of one base number are never negatives. The training set has 15,392 triplets, the benchmark 1,418. Part of the negative mining depends on stored LLM judge decisions, so the exact triplet lists cannot be regenerated from the recipe alone; they are available from the author on request.

File formats expected by the scripts:

- tales json: a list of `{"tale_id", "atu_id", "title", "text"}`
- triplet meta jsonl: `{"anchor_id", "pos_id", "neg_id", "anchor_atu", "pos_atu", "neg_atu"}` per line
- triplet text jsonl (optional, positionally aligned with the meta file): `{"anchor_text", "text_a", "text_b", "text_a_is_closer"}`

## Replication

All commands run from the repository root. `data/unique_tales.json` is the tales json for the 1,418 tales; `data/atu_union_train.meta.jsonl` and `data/atu_union_benchmark.meta.jsonl` are the triplet meta files.

1. Run the first extraction pass over the 1,417 tales.

```
python src/extract.py --tales data/unique_tales.json --prompt-doc prompts/extraction_prompt.md \
    --out data/extraction.jsonl --stats data/extraction_stats.json
```

2. Run the causal-only second pass, check it against the gold annotations, and merge.

```
python src/causal_pass.py --extractions data/extraction.jsonl --tales data/unique_tales.json \
    --prompt-doc prompts/causal_pass_prompt.md --out data/causal_pass.jsonl --stats data/causal_pass_stats.json
python src/eval_causal_pass.py --pred data/causal_pass.jsonl --json-out data/causal_gold_eval.json
python src/merge_causal.py --extractions data/extraction.jsonl --causal data/causal_pass.jsonl \
    --out data/extraction_causal.jsonl --stats data/merge_stats.json
```

The gold check must clear the three bars in `eval_causal_pass.py` (chain collapse below 65 percent, more than 15 percent multi-parent actions, more than 0.25 non-adjacent links per action). The merge reported 15,187 causal links over 14,779 actions on the full corpus (chain collapse 60.9 percent, 15.5 percent multi-parent).

3. Build the graphs and node-text encodings, and the type-disjoint split.

```
python src/build_graph_data.py --extractions data/extraction_causal.jsonl --tales data/unique_tales.json \
    --out-dir data/graphs --device cuda
python src/split_union.py --train-meta data/atu_union_train.meta.jsonl --out-dir data/graphs --seed 922
```

The split writes `union_train.jsonl` (12,426 triplets) and `union_val.jsonl` (1,861 triplets over whole held-out divisions). Model selection uses this validation split only; the benchmark is scored once per arm.

4. Train the 10 seeds with the flags in `configs/final.txt`.

```
for s in 7 111 222 333 444 555 666 922 2026 31337; do
  python src/train.py --data-dir data/graphs --train data/graphs/union_train.jsonl --val data/graphs/union_val.jsonl \
      --out-dir runs --run-id seed$s --seed $s --save-checkpoint checkpoints/ckpt_seed$s.pt \
      $(grep -v '^#' configs/final.txt)
done
```

5. Compute the text channels.

```
python src/encode_tales.py --tales data/unique_tales.json --out cache/embgemma_tale_cache.npz
python src/gemini_embed.py --tales data/unique_tales.json --cache cache/gemini_tale_cache.jsonl
```

6. Fit the fusion arms on the validation split, score every checkpoint on the benchmark, and run the interpretability suite.

```
python src/fusion.py --ckpt checkpoints/ckpt_seed7.pt --data-dir data/graphs \
    --train-meta data/graphs/union_train.jsonl --val-meta data/graphs/union_val.jsonl \
    --tales-json data/unique_tales.json --out-dir results --tag fusion --text-source gemini
python src/score.py --ckpt-glob "checkpoints/ckpt_seed*.pt" --data-dir data/graphs \
    --eval-meta data/atu_union_benchmark.meta.jsonl --tales-json data/unique_tales.json \
    --benchmark atu_union --out-dir results --tag score --fusion-arms results/fusion_arms.json \
    --text-sources gemini,embgemma --fusion-text gemini
python src/interpret.py --ckpt checkpoints/ckpt_seed7.pt --data-dir data/graphs \
    --eval-meta data/atu_union_benchmark.meta.jsonl --tales-json data/unique_tales.json \
    --benchmark atu_union --out-dir results --tag interpret --fusion-arms results/fusion_arms.json \
    --text-source gemini --knockout causes
```

`score.py` writes `score_table.md`, `score_aggregate.json` and `score_rows.jsonl`. The table reports the mean and standard deviation over the seeds and discloses the best seed separately; the headline is the mean. Expected values are in the table at the top of this file and in `results/`.

## Citation

```
@misc{sanjana2026interpretable,
  author = {Sanjana},
  title  = {Interpretable Narrative Similarity},
  year   = {2026},
  url    = {https://github.com/ndsanjana/interpretable-narrative-similarity}
}
```
