# BLiMP

Minimal-pair grammaticality evaluation. [`blimp_eval.py`](blimp_eval.py) scores
any directory of files in BLiMP's format for any model behind
[`evals.backends.LanguageModel`](../backends/base.py); the trainers call the
same function for their inline evaluations
([Guide 1.2](../../reproduce/icml2026/EVALUATION.md#12-benchmark-data-one-time-fetches),
[Guide 1.4](../../reproduce/icml2026/EVALUATION.md#14-blimp--table-2-blimp-column-table-5)).

[`fetch_data.py`](fetch_data.py) copies the
full set (67 paradigms × 1,000 pairs) from the original repository, at the
commit the ICML 2026 paper's evaluations used, into the git-ignored `data/`, and then writes the two inline evaluation
subsets the trainers read:

- `blimp_fast/` — 200 pairs per task: the inline evaluation set every
  pretraining run in the paper scored during training
- `blimp_really_fast/` — its first 10 rows per task: the validation set the
  LoRA fine-tuning runs tracked

Both are rebuilt from [`blimp_fast_ids.json`](blimp_fast_ids.json). The subset is from the BabyLM 2025 evaluation (`evaluation_data/fast_eval/blimp_fast` on its
[OSF node](https://osf.io/ryjfm/)) as of 2025-05-07; two of its
paradigms, `existential_there_quantifiers_2` and
`left_branch_island_echo_question`, have since been removed from that
folder, so the ids are the record.
