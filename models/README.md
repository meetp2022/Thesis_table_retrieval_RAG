# Models

This directory holds the trained model checkpoints used in the dissertation. The
weights themselves are **not committed to the repository**: the two fine-tuned
text encoders are roughly 420 MB each, which exceeds GitHub's 100 MB per-file
limit. They are also reproducible outputs of the training scripts rather than
source code, so this README documents how to regenerate them instead.

## Models used in the study

| Checkpoint | Produced by | Approx. size |
|---|---|---|
| `bge_finetuned/` | `scripts/finetune_bge.py` | ~420 MB |
| `mmqa_bge_finetuned/` | `scripts/finetune_bge_mmqa.py` | ~420 MB |
| `graph_encoder/` (WikiTQ, FinQA, TAT-QA) | `scripts/train_graph_encoder.py` | ~3 MB each |
| `mmqa_graph_encoder/` (intra, inter) | `scripts/train_mmqa_graph.py` | ~3 MB each |

The base text encoder is `BAAI/bge-base-en-v1.5` and the cross-encoder reranker is
`cross-encoder/ms-marco-MiniLM-L-6-v2`; both are downloaded automatically from the
Hugging Face Hub at run time and do not need to be stored here.

## Regenerating the checkpoints

From the repository root, with the environment installed (see the top-level
`README.md`) and the datasets downloaded (`python scripts/download_datasets.py`):

```bash
# Single-table text encoders (run per dataset: wikitq, finqa, tatqa)
python scripts/finetune_bge.py --dataset wikitq

# Single-table graph encoders
python scripts/train_graph_encoder.py --dataset wikitq

# Multi-table (MMQA) text encoder
python scripts/finetune_bge_mmqa.py

# Multi-table graph encoders (run for each edge set)
python scripts/train_mmqa_graph.py --edges intra
python scripts/train_mmqa_graph.py --edges inter
```

Each script writes its checkpoint into the corresponding sub-directory here. The
exact hyperparameters (learning rates, batch sizes, epochs, temperatures) are
fixed in the scripts and configs and are also listed in Appendix A of the
dissertation. All training uses a single fixed random seed.

## Hardware

Retrieval and fusion run on CPU. Encoder fine-tuning and graph training use a
single GPU; the single-table graph encoders train in a few minutes, while the
text-encoder fine-tuning is the most demanding step.

## Obtaining the exact weights

If you need the exact checkpoints used to produce the reported numbers rather
than regenerating them, please contact the author and they can be shared via an
external host (e.g. Hugging Face Hub or a download link).
