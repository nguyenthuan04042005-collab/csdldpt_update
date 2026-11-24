"""Validation checks and reasons for unavailable measurements."""

from itertools import groupby


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reason_intervals(reasons):
    spans = []
    offset = 0
    for reason, entries in groupby(reasons):
        length = sum(1 for _ in entries)
        if reason:
            spans.append([offset, offset + length, reason])
        offset += length
    return spans


def dimension_reasons(valid, names, info, cfg):
    reasons = {}
    length = info["longest_valid_pairs"]
    for name, ok in zip(names, valid):
        if ok:
            continue
        group = name[0]
        if group == "F":
            reason = "no_observed_F_cell"
        elif not info["valid_pairs"]:
            reason = "no_valid_G_pairs"
        elif group == "A":
            reason = "no_adjacent_valid_pairs"
        elif group == "C":
            reason = "undefined_pearson"
        elif group == "D":
            _, row_name, measure = name.split("_", 2)
            row = int(row_name[3:])
            if measure.startswith("quarter"):
                reason = "quarter_has_no_valid_pairs"
            elif measure.startswith("lag") and length < max(
                cfg["temporal"]["min_run"], 2 * int(measure[3:])
            ):
                reason = "run_too_short_for_lag"
            elif measure.startswith("peak"):
                reason = info["peak_status"][row]
            elif info["peak_status"][row] == "low_detrended_variance":
                reason = "low_detrended_variance"
            else:
                reason = "undefined_pearson"
        else:
            raise ArithmeticError(f"Unexpected unavailable feature: {name}")
        reasons[name] = reason
    return reasons
