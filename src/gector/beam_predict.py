"""
beam_predict.py — Beam search decoding for GECToR.

Strategy (agreed in design discussion)
---------------------------------------
Iteration axis  : GECToR's iterative correction rounds (like greedy predict()).
                  Each round runs a full forward pass per hypothesis and applies
                  the chosen tag sequence to produce the next hypothesis sentence.
                  This is the "tree depth" — NOT intra-sequence token steps.

Beam expansion  : At each round, every hypothesis is independently scored.
                  From each hypothesis we generate up to beam_size candidates:
                    1. The greedy tag sequence (always included).
                    2. Single-position swaps: for each word position, try the
                       top-k alternative tags (k = beam_size), keeping the swap
                       with the largest positive score delta.
                    3. Pairwise combinations of the top non-overlapping single
                       swaps (capped to avoid O(n²) blowup).
                  All candidates across all hypotheses are pooled and pruned
                  back to beam_size by score.

Scoring         : Sum of per-word log-probs (minimal-edit bias — penalises more
                  edits naturally since each non-KEEP tag has lower log-prob than
                  $KEEP for a well-calibrated model).  length_penalty adds a flat
                  bonus per non-KEEP edit to counteract over-conservatism.

Termination     : A hypothesis is marked finished when its best tag sequence is
                  all no-correction labels ($KEEP / <OOV> / <PAD>), or when
                  the sentence-level error probability is below min_error_prob.
                  The beam as a whole is retired when ALL hypotheses are finished,
                  or when n_iteration rounds have been exhausted.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set

from transformers import PreTrainedTokenizer

from .modeling import GECToR
from .predict import edit_src_by_tags, get_word_masks_from_word_ids
from .beam import Hypothesis, BeamState


# ── Forward pass + candidate generation ──────────────────────────────────────

def _score_hypotheses(
    model:             GECToR,
    tokenizer:         PreTrainedTokenizer,
    hypotheses:        List[Hypothesis],
    beam_size:         int,
    keep_confidence:   float,
    min_error_prob:    float,
    no_correction_ids: Set[int],
) -> Tuple[
    List[List[Tuple[float, List[str]]]],  # candidates per hyp: [(score, labels)]
    List[bool],                            # is_no_correction per hyp
]:
    """
    Run one forward pass over a flat list of hypothesis token sequences.

    For each hypothesis returns up to beam_size (score, label_sequence) candidates
    produced by:
      - the greedy sequence
      - top single-position swaps (best alternative tag at each word position)
      - top pairwise combinations of non-overlapping single swaps

    score = sum of word-level log-probs (lower = more edits = penalised).
    """
    srcs = [h.tokens for h in hypotheses]

    batch = tokenizer(
        srcs,
        return_tensors       = "pt",
        #max_length           = model.config.max_length,
        padding              = "max_length",
        truncation           = True,
        is_split_into_words  = True,
        add_special_tokens   = not model.config.is_official_model,
    )
    word_masks = torch.tensor(
        get_word_masks_from_word_ids(
            batch.word_ids,
            batch["input_ids"].size(0),
        )
    )
    batch      = {k: v.to(model.device) for k, v in batch.items()}
    word_masks = word_masks.to(model.device)

    with torch.no_grad():
        outputs   = model.forward(batch["input_ids"], batch["attention_mask"])
        log_probs = F.log_softmax(outputs.logits_labels, dim=-1)   # (B, L, V)
        prob_d    = F.softmax(outputs.logits_d,    dim=-1)          # (B, L, 2)

    keep_id  = model.config.label2id[model.config.keep_label]
    incor_id = model.config.d_label2id[model.config.incorrect_label]

    all_candidates    = []
    all_no_corrections = []

    for i in range(len(srcs)):
        lp = log_probs[i]   # (L, V)
        pd = prob_d[i]      # (L, 2)
        wm = word_masks[i]  # (L,)  1 at word-starts

        # Apply keep_confidence bias
        lp = lp.clone()
        lp[:, keep_id] += keep_confidence

        # Sentence-level error gate
        max_err_prob = (pd[:, incor_id] * wm).max().item()
        if max_err_prob < min_error_prob:
            all_no_corrections.append(True)
            all_candidates.append([(0.0, [model.config.keep_label] * lp.size(0))])
            continue

        # Word-start positions
        word_positions = wm.nonzero(as_tuple=True)[0].tolist()

        # ── Greedy base sequence ──────────────────────────────────────────
        greedy_ids    = lp.argmax(dim=-1)                    # (L,)
        greedy_labels = [model.config.id2label[idx.item()] for idx in greedy_ids]

        # Score = sum of word-start log-probs only (no length bias from padding)
        word_idx  = list(range(lp.size(0)))
        base_score = lp[word_idx, greedy_ids][wm.bool()].sum().item()

        candidates = [(base_score, greedy_labels)]

        # ── Single-position swaps ─────────────────────────────────────────
        # For each word position try top-(beam_size+1) alternatives and keep
        # the one with the largest positive delta vs greedy at that position.
        k_alt = min(beam_size + 1, lp.size(-1))
        topk  = lp.topk(k=k_alt, dim=-1)   # values/indices (L, k_alt)

        single_swaps: List[Tuple[float, List[str], int]] = []
        # (score, labels, changed_position)

        for pos in word_positions:
            greedy_id_at_pos    = greedy_ids[pos].item()
            greedy_lp_at_pos    = lp[pos, greedy_id_at_pos].item()

            for ki in range(1, k_alt):   # skip ki=0 (that IS the greedy)
                alt_id = topk.indices[pos, ki].item()
                if alt_id == greedy_id_at_pos:
                    continue

                delta      = topk.values[pos, ki].item() - greedy_lp_at_pos
                alt_labels = greedy_labels.copy()
                alt_labels[pos] = model.config.id2label[alt_id]
                swap_score = base_score + delta
                single_swaps.append((swap_score, alt_labels, pos))

        # Keep only the best swap per position (highest score)
        best_swap_per_pos: Dict[int, Tuple[float, List[str], int]] = {}
        for swap in single_swaps:
            pos = swap[2]
            if pos not in best_swap_per_pos or swap[0] > best_swap_per_pos[pos][0]:
                best_swap_per_pos[pos] = swap

        top_single_swaps = sorted(
            best_swap_per_pos.values(), key=lambda x: x[0], reverse=True
        )
        candidates.extend([(s, l) for s, l, _ in top_single_swaps])

        # ── Pairwise combinations ─────────────────────────────────────────
        # Combine top non-overlapping single swaps.
        # Cap at top-8 to keep O(n²) manageable on long sentences.
        pool = top_single_swaps[:8]
        for ii in range(len(pool)):
            for jj in range(ii + 1, len(pool)):
                score_i, labels_i, pos_i = pool[ii]
                score_j, labels_j, pos_j = pool[jj]

                if pos_i == pos_j:
                    continue   # same position — can't combine

                # Combined score: base + delta_i + delta_j
                combined_score  = base_score \
                                  + (score_i - base_score) \
                                  + (score_j - base_score)
                combined_labels = greedy_labels.copy()
                combined_labels[pos_i] = labels_i[pos_i]
                combined_labels[pos_j] = labels_j[pos_j]
                candidates.append((combined_score, combined_labels))

        # ── Prune to beam_size ────────────────────────────────────────────
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:beam_size]

        # no_correction: best candidate is entirely no-correction labels
        best_labels = candidates[0][1]
        no_corr = all(
            model.config.label2id.get(lbl, -1) in no_correction_ids
            for lbl in best_labels
            if lbl != model.config.label_pad_token
        )
        all_no_corrections.append(no_corr)
        all_candidates.append(candidates)

    return all_candidates, all_no_corrections


# ── Main beam search loop ─────────────────────────────────────────────────────

def beam_predict(
    model:           GECToR,
    tokenizer:       PreTrainedTokenizer,
    srcs:            List[str],
    encode:          Dict,
    decode:          Dict,
    beam_size:       int   = 4,
    keep_confidence: float = 0.0,
    min_error_prob:  float = 0.0,
    n_iteration:     int   = 5,
    batch_size:      int   = 64,
    length_penalty:  float = 0.0,
) -> List[str]:
    """
    Beam search over GECToR iterative correction rounds.

    Each "step" in the beam tree is one full GECToR iteration (forward pass +
    tag application), not one token.  The beam maintains beam_size full-sentence
    hypotheses per input sentence.

    Args:
        length_penalty: Flat score bonus per non-KEEP edit.  Counteracts the
                        natural conservatism of log-prob-sum scoring.  Try
                        small positive values (0.1–0.5) if the beam is too
                        conservative.
    Returns:
        List of corrected sentences, one per input.
    """
    no_correction_ids: Set[int] = {
        model.config.label2id[l]
        for l in ["$KEEP", "<OOV>", "<PAD>"]
        if l in model.config.label2id
    }

    # ── Initialise one BeamState per sentence ─────────────────────────────
    beam_states: List[BeamState] = [
        BeamState(
            beams      = [Hypothesis(tokens=["$START"] + src.split())],
            sentence_id = idx,
        )
        for idx, src in enumerate(srcs)
    ]

    final_outputs: Dict[int, str] = {}
    active_states = beam_states

    for iteration in range(n_iteration):
        if not active_states:
            break

        print(
            f"Iteration {iteration}: "
            f"{len(active_states)} sentences, "
            f"{sum(len(s.beams) for s in active_states)} total hypotheses"
        )

        # ── Flatten all active, unfinished hypotheses ─────────────────────
        # flat_hyps[k]  → the Hypothesis object
        # flat_index[k] → (state_idx_in_active_states, beam_idx)
        flat_hyps:  List[Hypothesis]         = []
        flat_index: List[Tuple[int, int]]    = []

        for s_idx, state in enumerate(active_states):
            for b_idx, hyp in enumerate(state.beams):
                if not hyp.is_finished:
                    flat_hyps.append(hyp)
                    flat_index.append((s_idx, b_idx))

        # ── Forward passes in sub-batches ─────────────────────────────────
        all_candidates_flat:     List[List[Tuple[float, List[str]]]] = []
        all_no_corrections_flat: List[bool]                          = []

        for start in range(0, len(flat_hyps), batch_size):
            chunk = flat_hyps[start: start + batch_size]
            cands, no_corrs = _score_hypotheses(
                model, tokenizer, chunk,
                beam_size, keep_confidence,
                min_error_prob, no_correction_ids,
            )
            all_candidates_flat.extend(cands)
            all_no_corrections_flat.extend(no_corrs)

        # ── Expand and pool candidates per state ──────────────────────────
        # Collect ALL new candidate hypotheses for each state, then prune
        # globally to beam_size.  This is the key difference from before:
        # we pool across parent hypotheses, not just within each parent.
        state_candidate_pool: Dict[int, List[Hypothesis]] = {
            i: [] for i in range(len(active_states))
        }

        for flat_i, (s_idx, b_idx) in enumerate(flat_index):
            parent   = active_states[s_idx].beams[b_idx]
            no_corr  = all_no_corrections_flat[flat_i]
            cands    = all_candidates_flat[flat_i]

            if no_corr:
                # Mark this hypothesis finished — no further edits needed
                finished = parent.clone()
                finished.is_finished = True
                state_candidate_pool[s_idx].append(finished)
                continue

            for score_delta, labels in cands:
                child = parent.clone()

                # Apply tag sequence to produce next token sequence
                edited = edit_src_by_tags(
                    [child.tokens], [labels], encode, decode
                )[0]

                # Length penalty: reward hypotheses that make edits
                n_edits = sum(
                    1 for lbl in labels
                    if model.config.label2id.get(lbl, -1) not in no_correction_ids
                )
                child.tokens      = edited
                child.score      += score_delta + length_penalty * n_edits
                child.tag_history.append(labels)
                state_candidate_pool[s_idx].append(child)

        # ── Prune each state to beam_size, retire finished states ─────────
        next_active_states: List[BeamState] = []

        for s_idx, state in enumerate(active_states):
            pool = state_candidate_pool[s_idx]

            if not pool:
                # Safety: keep existing beams unchanged if expansion failed
                pool = state.beams

            # Global prune: best beam_size hypotheses regardless of parent
            pool.sort(key=lambda h: h.score, reverse=True)
            state.beams = pool[:beam_size]

            if all(h.is_finished for h in state.beams):
                # All beams converged — emit best hypothesis
                best = state.best()
                final_outputs[state.sentence_id] = (
                    " ".join(best.tokens).replace("$START ", "")
                )
            else:
                next_active_states.append(state)

        active_states = next_active_states

    # ── Emit anything still active (hit iteration limit) ──────────────────
    for state in active_states:
        best = state.best()
        final_outputs[state.sentence_id] = (
            " ".join(best.tokens).replace("$START ", "")
        )

    return [final_outputs[i] for i in range(len(srcs))]