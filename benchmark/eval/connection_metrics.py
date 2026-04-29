from __future__ import annotations

from itertools import combinations
from typing import Any
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PinPair = Tuple[str, str]


def _safe_pins(pins: Iterable[object]) -> List[str]:
    if isinstance(pins, str):
        token = pins.strip()
        return [token] if token else []

    if not isinstance(pins, Iterable):
        return []

    return [str(p).strip() for p in pins if str(p).strip()]


def extract_pin_pairs(nets: Sequence[Dict[str, object]]) -> Set[PinPair]:
    """
    Convert net list into an unordered pin-pair set.
    """
    pairs: Set[PinPair] = set()
    for net in nets or []:
        if not isinstance(net, dict):
            continue
        pins = sorted(set(_safe_pins(net.get("pins", []))))
        for a, b in combinations(pins, 2):
            pairs.add((a, b))
    return pairs


def compute_connection_rp(
    gt_nets: Sequence[Dict[str, object]],
    gen_nets: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    """
    Compute connection recall/precision from pin-pair overlap.
    """
    gt_pairs = extract_pin_pairs(gt_nets)
    gen_pairs = extract_pin_pairs(gen_nets)

    tp = len(gt_pairs & gen_pairs)
    fp = len(gen_pairs - gt_pairs)
    fn = len(gt_pairs - gen_pairs)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    return recall, precision


def evaluate_against_variants(
    gt: Dict[str, object],
    gen_nets: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """
    Evaluate generated nets against primary and alternative GT variants.

    Returns the best variant by maximizing min(recall, precision), then recall,
    then precision.
    """
    if not isinstance(gen_nets, list):
        gen_nets = []

    variants: List[Tuple[str, Sequence[Dict[str, object]]]] = []
    primary = gt.get("nets", [])
    variants.append(("primary", primary if isinstance(primary, list) else []))

    alts: Any = gt.get("alternative_nets_sets", [])
    if not isinstance(alts, list):
        alts = []

    for idx, alt in enumerate(alts, start=1):
        if not isinstance(alt, dict):
            continue
        desc = str(alt.get("description", "")).strip() or f"alternative_{idx}"
        alt_nets = alt.get("nets", [])
        variants.append((desc, alt_nets if isinstance(alt_nets, list) else []))

    best = {
        "variant": "primary",
        "recall": 0.0,
        "precision": 0.0,
        "score": -1.0,
    }

    for label, variant_nets in variants:
        recall, precision = compute_connection_rp(variant_nets, gen_nets)
        score = min(recall, precision)
        if (
            score > best["score"]
            or (score == best["score"] and recall > best["recall"])
            or (
                score == best["score"]
                and recall == best["recall"]
                and precision > best["precision"]
            )
        ):
            best = {
                "variant": label,
                "recall": recall,
                "precision": precision,
                "score": score,
            }

    return best


def compute_best_connection_rp(
    gt: Dict[str, object],
    gen_nets: Sequence[Dict[str, object]],
) -> Tuple[float, float]:
    """
    Backward-compatible wrapper returning only (recall, precision).
    """
    best = evaluate_against_variants(gt, gen_nets)
    return float(best["recall"]), float(best["precision"])
