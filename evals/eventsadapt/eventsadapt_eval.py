#!/usr/bin/env python
"""Minimal-pair evaluation: BLiMP, its subsets, and any directory of files in its format.

Every ``.jsonl`` file at the top level of the input directory is a paradigm;
each line holds ``sentence_good`` / ``sentence_bad`` and, optionally,
``field``, ``linguistics_term`` and ``UID`` (a file without them counts as
"supplemental", named after the file). A pair is correct when the model
assigns the grammatical sentence the higher log-probability. Everything is
scored over the temperature grid at once; the reported temperature is the one
with the highest mean per-paradigm accuracy (first on ties), and T=1.0 is
reported beside it.

Library::

    from evals.blimp.blimp_eval import evaluate_blimp
    result = evaluate_blimp(backend, "evals/blimp/data")   # summary / rows / protocol

Command line (from the repository root)::

    python evals/blimp/blimp_eval.py --input_path evals/blimp/data --output_dir OUT \\
        --backend gptbert --backend-arg checkpoint=lm/gpt-bert/trained_models/<ckpt>.bin [--predict]

writes ``OUT/<backend name>/<input dir name>/{best_temperature_report.txt,temperature_1_report.txt}``
and, with ``--predict``, the chosen sentence per pair at T=1.0 and at the best temperature.
``--record PATH`` (with or instead of ``--output_dir``) writes the metrics the trainers log
(``training_metrics``) as one JSON: the best temperature, the average and per-field accuracies
at the best temperature and at T=1.0, and ``per_uid``, the per-paradigm accuracies at the best
temperature. ``--record_prefix P`` writes the two reports as ``P_best_temperature_report.txt``
and ``P_temperature_1_report.txt``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

import torch

import pandas as pd
import numpy as np
from sklearn import metrics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evals.backends import LanguageModel  # noqa: E402
from evals.common import TEMPERATURE_INDEX_1, TEMPERATURE_STEP, add_backend_arguments, backend_from_args, temperature_grid  # noqa: E402


def load_sentences(data_dir) -> List[Dict]:
    """Every pair of every top-level ``.jsonl`` file, in file order (files sorted by name)."""
    data_dir = pathlib.Path(data_dir)
    sentences = pd.DataFrame()
    for path in sorted(p for p in data_dir.iterdir() if p.suffix == ".csv"):
        new_sentences = pd.read_csv(path)
        sentences = pd.concat([sentences, new_sentences])
    return sentences[:50]


@torch.no_grad()
def evaluate_eventsadapt(backend: LanguageModel, data_dir, temperatures: Optional[Sequence[float]] = None,
                   progress: bool = False) -> Dict:
    """Score every pair under ``data_dir``; returns ``summary`` / ``rows`` / ``protocol``."""
    temps = torch.as_tensor(list(temperatures) if temperatures is not None else temperature_grid().tolist(),
                            dtype=torch.float32)
    n_temps = int(temps.numel())
    t1 = _index_of_one(temps)
    print(t1)
    sentences = load_sentences(data_dir)
    if sentences.empty:
        raise ValueError(f"no .csv sentences under {data_dir}")

    was_token = backend.encode("was")[0]
    by_token = backend.encode("by")[0]

    assert(backend.decode([was_token]) == "was")
    assert(backend.decode([by_token]) == "by")
    
    iterator = sentences.iterrows()

    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="sentences", total=sentences.shape[0])

    rows_mean_logprobs = []
    rows_mean_logprobs_std = []

    for _, row in iterator:
        encoded = backend.encode(row["Sentence"])

        scores = backend.sequence_pen_logprobs([encoded], temps.tolist())
        rows_mean_logprobs.append(scores.tolist()[0])

        if row["Voice"] == "passive":
            was_idx = encoded.index(was_token)
            by_idx = encoded.index(by_token)
            indices = list(range(len(encoded)))
            indices.remove(was_idx)
            indices.remove(by_idx)

            scores = backend.sequence_pen_logprobs([encoded], temps.tolist(), positions=[indices])
            rows_mean_logprobs_std.append(scores.tolist()[0])
        else:
            rows_mean_logprobs_std.append(scores.tolist()[0])

    sentences["logprobs_by_temperature"] = rows_mean_logprobs
    sentences["logprobs_std_by_temperature"] = rows_mean_logprobs_std

    return (sentences, temps.tolist(), t1)


def _accuracies(correct: Counter, total: Counter) -> Dict[str, float]:
    return OrderedDict((name, correct[name] / total[name] * 100.0) for name in total)


def _index_of_one(temps: torch.Tensor) -> int:
    return int((temps - 1.0).abs().argmin().item())


def _is_grid(temps: torch.Tensor) -> bool:
    return temps.numel() == 61 and torch.allclose(temps.clamp(min=TEMPERATURE_STEP / 2),
                                                  torch.arange(61, dtype=torch.float32).clamp(min=0.5) * TEMPERATURE_STEP,
                                                  atol=1e-6)


def training_metrics(result: pd.DataFrame, temps, t1) -> Dict[str, float]:
    print(t1)
    return result



# ---------------------------------------------------------------------------
# report and record writers
# ---------------------------------------------------------------------------
# def report_text(result: Dict, temperature_index: int, printed_temperature: float) -> str:
#     """The report for one temperature: per-field, per-term and per-paradigm accuracies, then the average."""
#     lines = [f"TEMPERATURE: {printed_temperature:.2f}", ""]
#     for key, title in REPORT_SECTIONS:
#         lines.append(f"### {title}")
#         lines.extend(f"{name}: {value:.2f}" for name, value in result["accuracy_by_temperature"][key][temperature_index].items())
#         lines.append("")
#     lines += ["### AVERAGE ACCURACY", f"{result['average_accuracy_by_temperature'][temperature_index]:.2f}", ""]
#     return "\n".join(lines) + "\n"


# def parse_report(text: str) -> Dict:
#     """Inverse of ``report_text`` (values only, order-insensitive)."""
#     out: Dict = {"sections": {}}
#     section = None
#     for line in text.splitlines():
#         if line.startswith("TEMPERATURE: "):
#             out["temperature"] = float(line.split(": ", 1)[1])
#         elif line.startswith("### "):
#             section = line[4:]
#             out["sections"][section] = {}
#         elif line.strip() and section == "AVERAGE ACCURACY":
#             out["average"] = float(line)
#         elif line.strip() and section:
#             name, _, value = line.rpartition(": ")
#             out["sections"][section][name] = float(value)
#     return out


# def write_outputs(result: Dict, out_dir, predict: bool = False) -> pathlib.Path:
#     out_dir = pathlib.Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
#     (out_dir / "best_temperature_report.txt").write_text(report_text(result, best, result["summary"]["blimp/best_temperature"]))
#     (out_dir / "temperature_1_report.txt").write_text(report_text(result, t1, 1.0))
#     if predict:
#         (out_dir / "predictions.json").write_text(json.dumps(predictions(result, t1)))
#         (out_dir / "predictions_at_best_temperature.json").write_text(json.dumps(predictions(result, best)))
#     return out_dir


# def write_reports(result: Dict, prefix) -> list:
#     """The two reports as ``<prefix>_best_temperature_report.txt`` and ``<prefix>_temperature_1_report.txt``."""
#     prefix = pathlib.Path(prefix)
#     prefix.parent.mkdir(parents=True, exist_ok=True)
#     best, t1 = result["protocol"]["best_temperature_index"], result["protocol"]["temperature_1_index"]
#     paths = [prefix.parent / f"{prefix.name}_best_temperature_report.txt",
#              prefix.parent / f"{prefix.name}_temperature_1_report.txt"]
#     paths[0].write_text(report_text(result, best, result["summary"]["blimp/best_temperature"]))
#     paths[1].write_text(report_text(result, t1, 1.0))
#     return paths


def write_record(metrics, path) -> pathlib.Path:
    """``training_metrics`` as one JSON: the scalars at the top level, ``per_uid`` nested, keys sorted."""
    metrics.to_csv(path)
    return path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", required=True, type=pathlib.Path, help="directory of .jsonl minimal-pair files")
    parser.add_argument("--output_dir", type=pathlib.Path,
                        help="run directory: reports under <output_dir>/<backend name>/<input dir name>/")
    parser.add_argument("--record", type=pathlib.Path,
                        help="write the metrics as one JSON: best temperature, average and per-field "
                             "accuracies at the best temperature and at T=1.0, per-paradigm accuracies "
                             "at the best temperature")
    parser.add_argument("--no_progress", action="store_true")
    add_backend_arguments(parser)
    args = parser.parse_args(argv)
    if args.output_dir is None and args.record is None and args.record_prefix is None:
        parser.error("give --output_dir, --record and/or --record_prefix")
    backend = backend_from_args(args)

    result, temps, t1 = evaluate_eventsadapt(backend, args.input_path, progress=not args.no_progress)
    metrics = training_metrics(result, temps, t1)
    written = []
    # if args.output_dir is not None:
    #     written.append(write_outputs(result, args.output_dir / backend.name / args.input_path.stem, predict=args.predict))
    if args.record is not None:
        written.append(write_record(metrics, args.record))
    # if args.record_prefix is not None:
    #     written.extend(write_reports(result, args.record_prefix))
    print(metrics)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
