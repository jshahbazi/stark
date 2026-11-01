"""Bounded, in-memory generation estimates using content-free measurements.

All durations start at native generation, after queueing, loading and prompt/image
preparation. Ranges are empirical estimates, never deadlines or guaranteed bounds.
"""

import math
from collections import OrderedDict, deque
from dataclasses import dataclass


def nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def progress_frame(value, previous=None):
    """Validate native progress without accepting content-bearing extra fields."""
    if (
        not isinstance(value, dict)
        or set(value)
        != {"event", "input_tokens", "sampled_tokens", "output_token_budget", "elapsed_seconds", "prefill_seconds"}
        or value["event"] != "progress"
    ):
        return None
    for key in ("input_tokens", "sampled_tokens", "output_token_budget"):
        if type(value[key]) is not int or not 0 <= value[key] <= 0xFFFFFFFF:
            return None
    if value["output_token_budget"] == 0 or value["sampled_tokens"] > value["output_token_budget"]:
        return None
    elapsed, prefill = value["elapsed_seconds"], value["prefill_seconds"]
    if not nonnegative(elapsed):
        return None
    if value["sampled_tokens"] == 0:
        if prefill is not None:
            return None
    elif not nonnegative(prefill) or prefill > elapsed:
        return None
    if previous is not None and (
        any(value[key] != previous[key] for key in ("input_tokens", "output_token_budget"))
        or value["sampled_tokens"] < previous["sampled_tokens"]
        or elapsed < previous["elapsed_seconds"]
        or (previous["prefill_seconds"] is not None and prefill != previous["prefill_seconds"])
    ):
        return None
    return dict(value)


@dataclass(frozen=True)
class Measurement:
    input_tokens: int
    sampled_tokens: int
    prefill_seconds: float
    seconds_per_token: float | None
    capped: bool


class GenerationEstimates:
    """Isolate model-load/settings profiles and retain only counts and durations."""

    def __init__(self, profiles=16, samples=24):
        self.profiles = profiles
        self.samples = samples
        self.history = OrderedDict()

    def record(self, key, reply):
        if "error" in reply:
            return
        input_tokens = reply.get("input_tokens")
        sampled = reply.get("sampled_tokens", reply.get("output_tokens"))
        prefill, decode = reply.get("prefill_seconds"), reply.get("decode_seconds")
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or type(sampled) is not int
            or sampled < 1
            or not nonnegative(prefill)
            or prefill == 0
        ):
            return
        per_token = decode / (sampled - 1) if sampled > 1 and nonnegative(decode) and decode > 0 else None
        self.history.setdefault(key, deque(maxlen=self.samples)).append(
            Measurement(input_tokens, sampled, prefill, per_token, reply.get("finish_reason") in {"length", "context"})
        )
        self.history.move_to_end(key)
        while len(self.history) > self.profiles:
            self.history.popitem(last=False)

    def estimate(self, key, progress):
        count, budget = progress["sampled_tokens"], progress["output_token_budget"]
        elapsed, prefill = progress["elapsed_seconds"], progress["prefill_seconds"]
        inputs = progress["input_tokens"]
        # Compare similar prompt sizes; linear scaling becomes misleading across
        # very different allocation/context regimes even within one model load.
        samples = [
            sample
            for sample in self.history.get(key, ())
            if 0.5 <= max(inputs, 1) / max(sample.input_tokens, 1) <= 2.0
        ]
        ttft_low = ttft_high = remaining_low = remaining_high = None
        predicted_prefill = [
            sample.prefill_seconds * max(inputs, 1) / max(sample.input_tokens, 1) for sample in samples
        ]
        if prefill is None and predicted_prefill:
            ttft_low = min(predicted_prefill) * 0.65
            ttft_high = max(max(predicted_prefill) * 1.6, elapsed * 1.3 + 0.1)

        rates = [sample.seconds_per_token for sample in samples if sample.seconds_per_token is not None]
        if count > 1 and prefill is not None and elapsed > prefill:
            live = (elapsed - prefill) / (count - 1)
            # Gradually prefer this request's measured speed. A few early tokens
            # do not erase the variability observed on previous requests.
            weight = min(1.0, (count - 1) / 16)
            rates = [(1 - weight) * rate + weight * live for rate in rates] or [live]
        if rates:
            slow, fast = max(rates) * 1.35, min(rates) * 0.75
            if samples:
                lengths = [budget if sample.capped else min(budget, sample.sampled_tokens) for sample in samples]
                low_target = min(budget, max(count + 1, min(lengths) * 0.6))
                high_target = min(budget, max(count + max(4, count * 0.5), max(lengths) * 1.8))
                remaining_low = max(0.0, low_target - max(1, count)) * fast
                remaining_high = max(0.0, high_target - max(1, count)) * slow
            elif count > 1:
                # With no response-length history, only the full remaining
                # token budget and current speed provide an honest broad range.
                remaining_low = 0.0
                remaining_high = max(0, budget - count) * slow
            if remaining_high is not None and prefill is None:
                if ttft_low is None:
                    remaining_low = remaining_high = None
                else:
                    remaining_low += max(0.0, ttft_low - elapsed)
                    remaining_high += max(0.0, ttft_high - elapsed)
        return {
            "phase": "prefill" if prefill is None else "decode",
            "sampled_tokens": count,
            "output_token_budget": budget,
            "elapsed_seconds": elapsed,
            "ttft_seconds": prefill,
            "ttft_lower_seconds": ttft_low,
            "ttft_upper_seconds": ttft_high,
            "remaining_lower_seconds": remaining_low,
            "remaining_upper_seconds": remaining_high,
            "calibrated": bool(samples),
        }
