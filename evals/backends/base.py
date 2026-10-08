"""The language-model interface the evaluators are written against.

A backend wraps one model and its tokenizer and answers three kinds of
question: how text maps to token ids, how likely each token is under the
model, and what the model's hidden states are. Everything else -- sequence
scores, span scores, temperature grids, accuracy bookkeeping, sharding across
processes -- belongs to the evaluators.

Conventions every backend must honour (``evals/backends/conformance.py``
checks them on a tiny model):

* ``encode`` / ``decode`` work on the caller's text without special tokens;
  ``encode_with_offsets`` maps every token to a character span of the text.
* ``token_logprobs(seqs, temperatures)`` returns, per input sequence, a
  float32 CPU tensor of shape ``[n_temperatures, len(seq)]`` holding the
  natural-log probability of each *input* token, with the model's logits
  divided by the temperature before the log-softmax. Special tokens, padding
  and batching are the backend's business; the caller's tokens come back in
  order, and a sequence scores the same alone or in any batch.
* ``hidden_states(seqs, layers)`` returns, per input sequence, a float32 CPU
  tensor ``[n_layers_selected, len(seq), width]`` at the caller's token
  positions (special tokens stripped, no padding rows).
"""

from __future__ import annotations

import abc
import importlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

Layers = Union[str, Sequence[int]]

# Short names for the backends shipped with this repository.
BACKEND_ALIASES: Dict[str, str] = {
    "gptbert": "evals.backends.gptbert:GPTBert",
}


class LanguageModel(abc.ABC):
    """One model plus its tokenizer, behind a family-agnostic interface."""

    #: a label for reports and records
    name: str = "model"
    #: the device forward passes run on
    device: torch.device = torch.device("cpu")
    #: the longest token sequence the model accepts, or None when unbounded
    max_seq_len: Optional[int] = None
    #: number of layers ``hidden_states`` can return
    n_layers: int = 0

    # --- text <-> ids -------------------------------------------------------
    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        """Token ids of ``text`` with no special tokens added."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Text for ``ids`` (special tokens dropped)."""

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Token ids and, per token, the ``[start, end)`` character span of ``text`` it covers.

        The default reconstructs spans by decoding the ids one at a time and
        locating each decoded piece in ``text``; a backend whose tokenizer
        provides offsets should override it.
        """
        ids = self.encode(text)
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        previous = ""
        for i in range(len(ids)):
            current = self.decode(ids[: i + 1])
            piece = current[len(previous):] if current.startswith(previous) else current
            previous = current
            piece = piece.strip()
            if not piece:
                offsets.append((cursor, cursor))
                continue
            found = text.find(piece, cursor)
            if found < 0:  # normalisation changed the surface; take the next non-space run
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                found = cursor
            offsets.append((found, found + len(piece)))
            cursor = found + len(piece)
        return ids, offsets

    # --- scoring --------------------------------------------------------------
    @abc.abstractmethod
    def token_logprobs(self, seqs: Sequence[Sequence[int]], temperatures: Sequence[float] = (1.0,),
                       prefix_lens: Optional[Sequence[int]] = None,
                       positions: Optional[Sequence[Sequence[int]]] = None) -> List[torch.Tensor]:
        """Per sequence a ``[n_temperatures, len(seq)]`` tensor of token log-probabilities.

        Two optional hints: ``prefix_lens`` gives, per sequence, the number of
        leading tokens that are context rather than target, which a backend
        may use to condition its scoring; ``positions`` gives, per sequence,
        the token indices the caller will read, so a backend that scores
        positions one at a time can skip the rest -- the returned tensor keeps
        its full shape and holds 0 at the positions not asked for. Backends
        without such notions ignore the hints.
        """

    def sequence_logprobs(self, seqs: Sequence[Sequence[int]], temperatures: Sequence[float] = (1.0,),
                          prefix_lens: Optional[Sequence[int]] = None,
                          positions: Optional[Sequence[Sequence[int]]] = None) -> torch.Tensor:
        """``[n_seqs, n_temperatures]`` sums of ``token_logprobs`` (over ``positions`` when given)."""
        per_token = self.token_logprobs(seqs, temperatures, prefix_lens, positions)
        return torch.stack([t.sum(dim=-1) for t in per_token], dim=0)

    def sequence_mean_logprobs(self, seqs: Sequence[Sequence[int]], temperatures: Sequence[float] = (1.0,),
                              prefix_lens: Optional[Sequence[int]] = None,
                              positions: Optional[Sequence[Sequence[int]]] = None) -> torch.Tensor:
            """``[n_seqs, n_temperatures]`` sums of ``token_logprobs`` (over ``positions`` when given)."""
            per_token = self.token_logprobs(seqs, temperatures, prefix_lens, positions)
            return torch.stack([t.mean(dim=-1) for t in per_token], dim=0)

    def sequence_pen_logprobs(self, seqs: Sequence[Sequence[int]], temperatures: Sequence[float] = (1.0,),
                                  prefix_lens: Optional[Sequence[int]] = None,
                                  positions: Optional[Sequence[Sequence[int]]] = None,
                                  L: float = 5.0, alpha: float = 0.8) -> torch.Tensor:
                """``[n_seqs, n_temperatures]`` sums of ``token_logprobs`` (over ``positions`` when given)."""
                per_token = self.token_logprobs(seqs, temperatures, prefix_lens, positions)
                return torch.stack([t.sum(dim=-1) for t in per_token], dim=0) / ((L + len(seqs)) / (L + 1)) ** alpha

    # --- representations ------------------------------------------------------
    @abc.abstractmethod
    def hidden_states(self, seqs: Sequence[Sequence[int]], layers: Layers = "final",
                      add_special_tokens: bool = True) -> List[torch.Tensor]:
        """Per sequence a ``[n_layers_selected, len(seq), width]`` tensor.

        ``layers`` is ``"final"`` or a sequence of layer indices (negative
        counts from the last). With ``add_special_tokens`` the sequence is
        contextualised the way the model is normally run (for the released
        models: a leading ``<s>``); without, the bare ids are fed.
        """

    # --- construction ---------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def from_args(cls, args: Dict[str, str], device: Optional[torch.device] = None) -> "LanguageModel":
        """Build a backend from ``--backend-arg key=value`` pairs."""

    @classmethod
    def argument_help(cls) -> str:
        """One line per accepted ``--backend-arg`` key."""
        return ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, device={self.device})"


def parse_backend_args(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    """``["k=v", ...]`` -> ``{"k": "v"}``; a pair without ``=`` is an error."""
    out: Dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"--backend-arg expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value
    return out


def resolve_backend_class(spec: str):
    """``alias`` or ``module:Class`` -> the class."""
    target = BACKEND_ALIASES.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown backend {spec!r}; use an alias {sorted(BACKEND_ALIASES)} or module:Class")
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    if not (isinstance(klass, type) and issubclass(klass, LanguageModel)):
        raise TypeError(f"{target} is not a LanguageModel subclass")
    return klass


def load_backend(spec: str, args: Dict[str, str], device: Optional[torch.device] = None) -> LanguageModel:
    """Instantiate the backend named by ``spec`` from its ``key=value`` options."""
    return resolve_backend_class(spec).from_args(dict(args), device=device)
