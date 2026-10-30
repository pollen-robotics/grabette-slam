"""Pair saved left/depth pictures by their common capture timeline.

Sequence counters identify pictures within a stream; counters from separate
streams need not have the same origin. Never use a cross-stream seq join.
"""

from bisect import bisect_left, bisect_right
import math


DEFAULT_PAIR_TOLERANCE_MS = 1000.0 / 60.0


def _validated_rows(samples, stream):
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{stream}: expected a non-empty timestamp sample list")
    rows = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{stream}[{index}]: expected a timestamp object")
        stamp = sample.get("host_ms")
        seq = sample.get("seq")
        if (not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
                or not math.isfinite(stamp)):
            raise ValueError(f"{stream}[{index}]: missing/invalid host_ms")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError(f"{stream}[{index}]: missing/invalid seq")
        if rows and stamp <= rows[-1]["host_ms"]:
            raise ValueError(f"{stream}[{index}]: host_ms must be strictly increasing")
        if rows and seq <= rows[-1]["seq"]:
            raise ValueError(f"{stream}[{index}]: seq must be strictly increasing")
        rows.append({"frame_index": index, "seq": seq, "host_ms": float(stamp)})
    return rows


def pair_left_depth(left_samples, depth_samples,
                    max_delta_ms=DEFAULT_PAIR_TOLERANCE_MS):
    """Return a reproducible, bounded, one-to-one matching in saved order.

    Values outside the depth stream's temporal coverage are rejected. For each
    remaining left picture, choose the closest still-unused depth picture
    within ``max_delta_ms``; an exact tie chooses the earlier depth index. The
    chosen depth indices strictly increase, so pictures cannot be reused or
    reordered. Rejected left pictures retain their original decoded indices.

    This is a pairing rule, not evidence of simultaneous sensor exposure.
    Legacy oakd_ sidecars have the same samples/host_ms/seq schema and are
    resolved by the caller; a missing common-clock timestamp is an error.
    """
    if (not isinstance(max_delta_ms, (int, float)) or isinstance(max_delta_ms, bool)
            or not math.isfinite(max_delta_ms) or max_delta_ms < 0):
        raise ValueError("max_delta_ms must be a finite nonnegative number")
    left = _validated_rows(left_samples, "left")
    depth = _validated_rows(depth_samples, "depth")
    depth_times = [row["host_ms"] for row in depth]
    pairs, rejected = [], []
    last_depth_index = -1
    for row in left:
        stamp = row["host_ms"]
        reason = None
        if stamp < depth_times[0] or stamp > depth_times[-1]:
            reason = "outside_depth_coverage"
        else:
            lower = max(last_depth_index + 1,
                        bisect_left(depth_times, stamp - max_delta_ms))
            upper = bisect_right(depth_times, stamp + max_delta_ms)
            if lower >= upper:
                reason = "no_unused_depth_within_tolerance"
            else:
                position = bisect_left(depth_times, stamp, lower, upper)
                candidates = (min(position, upper - 1), max(position - 1, lower))
                index = min(candidates, key=lambda i: (abs(depth_times[i] - stamp), i))
                selected = depth[index]
                last_depth_index = index
                pairs.append({
                    "left_frame_index": row["frame_index"],
                    "left_seq": row["seq"], "left_host_ms": stamp,
                    "depth_frame_index": index,
                    "depth_seq": selected["seq"],
                    "depth_host_ms": selected["host_ms"],
                    "residual_ms": selected["host_ms"] - stamp,
                })
        if reason:
            rejected.append({"left_frame_index": row["frame_index"],
                             "left_seq": row["seq"], "left_host_ms": stamp,
                             "reason": reason})
    used = {row["depth_frame_index"] for row in pairs}
    return {
        "schema_version": 1,
        "method": "common_host_ms_bounded_nearest_one_to_one",
        "max_delta_ms": float(max_delta_ms),
        "left_timestamp_count": len(left), "depth_timestamp_count": len(depth),
        "matched_count": len(pairs), "rejected_left_count": len(rejected),
        "max_absolute_residual_ms": max((abs(p["residual_ms"]) for p in pairs), default=None),
        "depth_reuse_count": 0,
        "pairs": pairs, "rejected_left": rejected,
        "unused_depth_frame_indices": [i for i in range(len(depth)) if i not in used],
    }
