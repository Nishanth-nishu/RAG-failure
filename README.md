# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked copy of the **GDPR (Regulation (EU) 2016/679, 88 pages)**:

| | Pipeline 1 – Baseline | Pipeline 2 – Re-ranking |
|---|---|---|
| Stage 1 | `all-MiniLM-L6-v2` bi-encoder, cosine, Top-10 | same Top-10 |
| Stage 2 | take Top-3 | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores 10 pairs → Top-3 |
| Generation | identical prompt → Answer 1 | identical prompt → Answer 2 |

The LLM is `Qwen2.5-0.5B-Instruct`, run locally on CPU with greedy decoding. If `GEMINI_API_KEY` is set, the notebook uses Gemini instead.

## Files

| Path | Content |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | **Main deliverable**: commented, executed notebook covering every step |
| `REPORT.md` | Detailed analysis report |
| `data/gdpr.pdf` | Source document (official EUR-Lex PDF) |
| `results/` | CSV/JSON outputs: chunks, rank shifts, LLM answers, metrics, latency, re-rank depth |
| `figures/` | Rank-shift slope chart and retrieval-metric bar chart |
| `build_notebook.py` | Script that generated the notebook cells |

## Run

```bash
pip install -r requirements.txt
# optional: download the LLM once into models/ (otherwise it is pulled from the HF hub on first run)
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

Runtime on a 13th-gen i5 CPU is about 10 minutes, most of it in the full-corpus cross-encoder experiment and LLM generation.
"# RAG-failure" 
