# Graph-Augmented Table Retrieval in RAG: A Cross-Regime Ablation

Implementation for the MSc dissertation *"Graph-Augmented Table Retrieval in Retrieval-Augmented Generation: A Cross-Regime Ablation of Where Relational Structure Helps"* (Gisma University of Applied Sciences, 2026).

This repository contains the complete, reproducible code for a controlled study of whether encoding a table's structure with a graph neural network improves retrieval for retrieval-augmented generation (RAG), across both single-table and multi-table settings.

## Summary of findings

The study holds every component of a table-retrieval pipeline fixed and varies only the graph component, so the marginal value of relational structure can be measured rather than assumed.

- **A graph encoder is statistically redundant** once a fine-tuned text encoder and a cross-encoder reranker are in place, on all three single-table benchmarks (WikiTableQuestions, FinQA, TAT-QA).
- **Inter-table foreign-key edges add nothing** over an intra-table-only graph on MMQA, a benchmark built to require cross-table reasoning (Δ = 0.005, p = 0.10).
- **A structure-blind reranker is actively harmful** to a strong retriever, while a **foreign-key-aware reranker recovers a large gain** (+0.151 full-set Recall@10, p < 10⁻⁴).
- Conclusion: for table retrieval, relational structure belongs in the **reranker's input**, not in a graph encoder's message passing.

## Repository structure

```
src/
  data/          dataset loaders (single-table + MMQA)
  graph/         table-to-graph construction, feature extraction, GraphSAGE encoder, training
  pipelines/     retrieval pipelines: text, graph-augmented, hybrid, image baseline, shared components
  evaluation/    single-table and multi-table metrics
  utils/         config and logging helpers
scripts/         data download, training, fine-tuning, evaluation, statistics, figures
configs/         experiment configuration files
tests/           unit tests
notebooks/       GPU training notebooks (Kaggle)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10. Retrieval and fusion run on CPU; encoder fine-tuning and graph training use a single GPU.

## Datasets

All datasets are public and loaded through the Hugging Face Hub:

| Dataset | Hub id |
|---|---|
| WikiTableQuestions | `stanfordnlp/wikitablequestions` |
| FinQA | `dreamerdeo/finqa` |
| TAT-QA | `next-tat/tat-qa` |
| MMQA (multi-table) | `table-benchmark/mmqa` |

> Note: the MMQA used here is the multi-table benchmark (Wu et al., ICLR 2025), distinct from the similarly named MultiModalQA dataset.

Download with:

```bash
python scripts/download_datasets.py
```

## Reproducing the results

### Single-table regime

```bash
# Fine-tune the text encoder, then evaluate each pipeline
python scripts/finetune_bge.py --dataset wikitq
python scripts/train_graph_encoder.py --dataset wikitq
python scripts/eval_hybrid_finetuned.py --dataset wikitq --n 1000
python scripts/bootstrap_ci.py --dataset wikitq
```

Repeat with `--dataset finqa` and `--dataset tatqa`.

### Multi-table regime (MMQA)

```bash
python scripts/build_mmqa_corpus.py
python scripts/build_mmqa_graph.py            # builds intra and inter edge sets
python scripts/finetune_bge_mmqa.py
python scripts/train_mmqa_graph.py --edges inter
python scripts/eval_mmqa_hybrid.py            # retrieval + reranking ablations
python scripts/bootstrap_mmqa.py              # paired bootstrap CIs + McNemar
```

### Figures and verification

```bash
python scripts/generate_figures_v2.py
python scripts/verify_all_results.py          # cross-checks every reported number against saved results
```

## Method at a glance

- **Text leg:** BGE-base-en-v1.5, fine-tuned with a multiple-negatives ranking loss.
- **Graph leg:** 2-layer GraphSAGE (774 → 256 → 768) over a cell/header/table-node graph; trained with a contrastive (InfoNCE) objective.
- **Fusion:** per-query z-normalised late score fusion, weight α tuned on development data.
- **Reranking:** MS-MARCO MiniLM cross-encoder; standard vs. a joint variant that appends each candidate's top foreign-key neighbour.
- **Statistics:** paired bootstrap confidence intervals (10,000 resamples) and McNemar's exact test.

## Citation

If you refer to this work, please cite the dissertation:

> Patel, M. (2026). *Graph-Augmented Table Retrieval in Retrieval-Augmented Generation: A Cross-Regime Ablation of Where Relational Structure Helps.* MSc dissertation, Gisma University of Applied Sciences.

## License

See [LICENSE](LICENSE).
