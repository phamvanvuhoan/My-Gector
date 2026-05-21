import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .modeling import GECToR, GECToRPredictionOutput
from .predict import edit_src_by_tags, get_word_masks_from_word_ids
from .beam import Hypothesis, BeamState
from transformers import PreTrainedTokenizer


def _score_hypotheses(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    hypotheses: List[Hypothesis],   # one per sentence (the tokens to score)
    beam_size: int,
    keep_confidence: float,
    min_error_prob: float,
    no_correction_ids: List[int],
) -> Tuple[
    List[List[Tuple[float, List[str]]]],  # top-K (delta_score, labels) per hyp
    List[bool]                             # is_no_correction per hyp
]:
    """
    Run one forward pass over the current hypothesis tokens.
    Returns top-K (score_delta, tag_sequence) candidates per hypothesis,
    plus a flag indicating whether each hypothesis needs no further edits.
    """
    srcs = [h.tokens for h in hypotheses]
    batch = tokenizer(
        srcs,
        return_tensors='pt',
        #max_length=model.config.max_length,
        padding='max_length',
        truncation=True,
        is_split_into_words=True,
        add_special_tokens=not model.config.is_official_model
    )
    word_masks = torch.tensor(
        get_word_masks_from_word_ids(
            batch.word_ids,
            batch['input_ids'].size(0)
        )
    )
    batch = {k: v.to(model.device) for k, v in batch.items()}
    word_masks = word_masks.to(model.device)

    with torch.no_grad():
        outputs = model.forward(
            batch['input_ids'],
            batch['attention_mask']
        )
        # (batch, seq_len, num_labels)
        log_probs = F.log_softmax(outputs.logits_labels, dim=-1)
        prob_d = F.softmax(outputs.logits_d, dim=-1)

    keep_id = model.config.label2id[model.config.keep_label]
    incor_id = model.config.d_label2id[model.config.incorrect_label]

    all_candidates = []   # List[List[(score_delta, labels)]]
    all_no_corrections = []

    for i in range(len(srcs)):
        word_ids_fn = batch['input_ids']  # we use word_ids from tokenizer
        lp = log_probs[i]          # (seq_len, num_labels)
        pd = prob_d[i]             # (seq_len, 2)

        # Apply keep_confidence bias (same as greedy predict())
        lp[:, keep_id] += keep_confidence

        # Sentence-level max error prob threshold
        wm = word_masks[i]
        max_err_prob = (pd[:, incor_id] * wm).max().item()
        if max_err_prob < min_error_prob:
            # Force all-KEEP: this hypothesis needs no correction
            all_no_corrections.append(True)
            all_candidates.append([(0.0, ['$KEEP'] * lp.size(0))])
            continue

        # --- Beam expansion ---
        # For efficiency we use the greedy sequence score as the base beam,
        # plus top-K alternatives found by perturbing each position.
        #
        # Strategy: compute the sentence-level score as the SUM of
        # per-token log-probs (only at word-start positions), then
        # expand by swapping one token to its 2nd-best prediction.
        # This gives O(seq_len * beam_size) candidates cheaply.

        word_positions = [
            j for j in range(lp.size(0))
            if batch.get('attention_mask', None) is not None
            # filter to real word-start tokens using word_masks
            and wm[j].item() == 1
        ]

        # Base greedy sequence
        greedy_ids = lp.argmax(dim=-1)  # (seq_len,)
        base_score = lp[
            range(lp.size(0)), greedy_ids
        ][wm.bool()].sum().item()
        greedy_labels = [
            model.config.id2label[idx.item()] for idx in greedy_ids
        ]

        candidates = [(base_score, greedy_labels)]

        # Expand: for each word position, try the 2nd-best tag
        top2 = lp.topk(k=min(2, lp.size(-1)), dim=-1)  # values, indices
        for pos in word_positions:
            if top2.indices[pos, 0].item() == top2.indices[pos, 1].item():
                continue
            alt_id = top2.indices[pos, 1].item()
            # Score = base_score - score_of_greedy_at_pos + score_of_alt
            delta = (
                top2.values[pos, 1].item()
                - top2.values[pos, 0].item()
            )
            alt_labels = greedy_labels.copy()
            alt_labels[pos] = model.config.id2label[alt_id]
            candidates.append((base_score + delta, alt_labels))

        # Keep only top beam_size candidates by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:beam_size]

        # Determine no_correction: True only if the BEST candidate is all-KEEP
        best_labels = candidates[0][1]
        no_corr = all(
            model.config.label2id.get(l, -1) in no_correction_ids
            for l in best_labels
        )
        all_no_corrections.append(no_corr)
        all_candidates.append(candidates)

    return all_candidates, all_no_corrections

def beam_predict(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    srcs: List[str],
    encode: Dict,
    decode: Dict,
    beam_size: int = 4,
    keep_confidence: float = 0.0,
    min_error_prob: float = 0.0,
    n_iteration: int = 5,
    batch_size: int = 64,          # reduce vs greedy; memory scales with beam_size
    length_penalty: float = 0.0,   # > 0 favours longer edits
) -> List[str]:
    """
    Beam search over GECToR edit sequences.

    For each sentence, maintains `beam_size` hypotheses across `n_iteration`
    rounds.  Each hypothesis is an independent token sequence.

    Args:
        length_penalty: Score bonus per non-KEEP edit applied.
                        Counteracts the model's natural conservatism.
    Returns:
        List of corrected sentences (one per input).
    """
    no_correction_ids = set(
        model.config.label2id[l]
        for l in ['$KEEP', '<OOV>', '<PAD>']
        if l in model.config.label2id
    )

    # Initialise one BeamState per sentence
    beam_states: List[BeamState] = [
        BeamState(
            beams=[Hypothesis(tokens=['$START'] + src.split())],
            sentence_id=idx
        )
        for idx, src in enumerate(srcs)
    ]
    # Track final outputs
    final_outputs: Dict[int, str] = {}

    active_states = beam_states  # states still being processed

    for iteration in range(n_iteration):
        if not active_states:
            break
        print(
            f"Iteration {iteration}: {len(active_states)} sentences, "
            f"{sum(len(s.beams) for s in active_states)} total beams"
        )

        # Flatten all active beams into a single batch
        # Track which (state_idx, beam_idx) each flat entry maps to
        flat_hyps: List[Hypothesis] = []
        flat_index: List[Tuple[int, int]] = []  # (state_idx, beam_idx)

        for s_idx, state in enumerate(active_states):
            for b_idx, hyp in enumerate(state.beams):
                if not hyp.is_finished:
                    flat_hyps.append(hyp)
                    flat_index.append((s_idx, b_idx))

        # Process in sub-batches to control GPU memory
        all_candidates_flat = []
        all_no_corrections_flat = []

        for start in range(0, len(flat_hyps), batch_size):
            chunk = flat_hyps[start: start + batch_size]
            cands, no_corrs = _score_hypotheses(
                model, tokenizer, chunk,
                beam_size, keep_confidence,
                min_error_prob, list(no_correction_ids)
            )
            all_candidates_flat.extend(cands)
            all_no_corrections_flat.extend(no_corrs)

        # Scatter results back and expand beams
        # Collect per-state new candidates
        state_new_beams: Dict[int, List[Hypothesis]] = {
            i: [] for i in range(len(active_states))
        }

        for flat_i, (s_idx, b_idx) in enumerate(flat_index):
            parent_hyp = active_states[s_idx].beams[b_idx]
            no_corr = all_no_corrections_flat[flat_i]
            candidates = all_candidates_flat[flat_i]

            if no_corr:
                finished = parent_hyp.clone()
                finished.is_finished = True
                state_new_beams[s_idx].append(finished)
                continue

            for score_delta, labels in candidates:
                child = parent_hyp.clone()

                # Apply edits to produce next token sequence
                edited = edit_src_by_tags(
                    [child.tokens],
                    [labels],
                    encode, decode
                )[0]

                # Length penalty: reward sentences that actually changed
                n_edits = sum(
                    1 for l in labels
                    if model.config.label2id.get(l, -1)
                    not in no_correction_ids
                )
                penalty = length_penalty * n_edits

                child.tokens = edited
                child.score += score_delta + penalty
                child.tag_history.append(labels)
                state_new_beams[s_idx].append(child)

        # Prune each state back to beam_size
        next_active_states = []
        for s_idx, state in enumerate(active_states):
            new_beams = state_new_beams[s_idx]
            if not new_beams:
                # Safety fallback: keep existing beams unchanged
                new_beams = state.beams

            # Sort and prune
            new_beams.sort(key=lambda h: h.score, reverse=True)
            state.beams = new_beams[:beam_size]

            # If ALL beams are finished, emit this sentence
            if all(h.is_finished for h in state.beams):
                best = state.best()
                final_outputs[state.sentence_id] = (
                    ' '.join(best.tokens).replace('$START ', '')
                )
            else:
                next_active_states.append(state)

        active_states = next_active_states

    # Emit anything still active (hit iteration limit)
    for state in active_states:
        best = state.best()
        final_outputs[state.sentence_id] = (
            ' '.join(best.tokens).replace('$START ', '')
        )

    return [final_outputs[i] for i in range(len(srcs))]