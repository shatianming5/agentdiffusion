"""Emergent phenomena detection in generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmergenceEvent:
    event_type: str   # "flash_crash", "bubble", "liquidity_crisis", "herding"
    start_step: int
    end_step: int
    severity: float
    description: str


def detect_flash_crashes(
    prices: np.ndarray,
    threshold_pct: float = 5.0,
    recovery_window: int = 20,
) -> list[EmergenceEvent]:
    """Detect flash crash events: rapid price drop > threshold followed by partial recovery."""
    events = []
    returns = np.diff(prices) / (prices[:-1] + 1e-10) * 100

    for i in range(len(returns)):
        if returns[i] < -threshold_pct:
            # Check for recovery
            end = min(i + recovery_window, len(prices) - 1)
            recovery = (prices[end] - prices[i + 1]) / (prices[i] - prices[i + 1] + 1e-10)

            events.append(EmergenceEvent(
                event_type="flash_crash",
                start_step=i,
                end_step=end,
                severity=abs(returns[i]),
                description=f"Price drop {returns[i]:.1f}%, recovery {recovery*100:.0f}%",
            ))

    return events


def detect_bubbles(
    prices: np.ndarray,
    window: int = 50,
    bubble_threshold: float = 2.0,
) -> list[EmergenceEvent]:
    """Detect bubble-crash cycles: sustained rise followed by sharp reversal."""
    events = []
    if len(prices) < window * 2:
        return events

    for i in range(window, len(prices) - window):
        # Measure cumulative return in window before and after
        pre_return = (prices[i] - prices[i - window]) / (prices[i - window] + 1e-10)
        post_return = (prices[i + window] - prices[i]) / (prices[i] + 1e-10)

        # Bubble: strong positive pre + strong negative post
        if pre_return > bubble_threshold * 0.01 and post_return < -bubble_threshold * 0.005:
            events.append(EmergenceEvent(
                event_type="bubble",
                start_step=i - window,
                end_step=i + window,
                severity=pre_return - post_return,
                description=f"Rise {pre_return*100:.1f}% then fall {post_return*100:.1f}%",
            ))

    # Deduplicate overlapping events
    return _deduplicate_events(events)


def detect_liquidity_crisis(
    spreads: np.ndarray,
    threshold_multiplier: float = 3.0,
) -> list[EmergenceEvent]:
    """Detect liquidity crises: spread exceeds N× its historical average."""
    events = []
    if len(spreads) < 50:
        return events

    rolling_mean = np.convolve(spreads, np.ones(50)/50, mode="valid")
    offset = len(spreads) - len(rolling_mean)

    for i, (s, m) in enumerate(zip(spreads[offset:], rolling_mean)):
        if s > threshold_multiplier * m:
            events.append(EmergenceEvent(
                event_type="liquidity_crisis",
                start_step=i + offset,
                end_step=i + offset,
                severity=s / (m + 1e-10),
                description=f"Spread {s/m:.1f}× historical average",
            ))

    return _deduplicate_events(events)


def detect_herding(
    volumes: np.ndarray,
    volatilities: np.ndarray,
    correlation_threshold: float = 0.8,
    window: int = 20,
) -> list[EmergenceEvent]:
    """Detect herding: synchronized volume and volatility spikes."""
    events = []
    if len(volumes) < window:
        return events

    for i in range(window, len(volumes)):
        v_window = volumes[i-window:i]
        vol_window = volatilities[i-window:i]

        if np.std(v_window) < 1e-10 or np.std(vol_window) < 1e-10:
            continue

        corr = np.corrcoef(v_window, vol_window)[0, 1]
        v_spike = v_window[-1] / (np.mean(v_window) + 1e-10)

        if corr > correlation_threshold and v_spike > 2.0:
            events.append(EmergenceEvent(
                event_type="herding",
                start_step=i - window,
                end_step=i,
                severity=corr * v_spike,
                description=f"Volume-volatility corr={corr:.2f}, volume spike={v_spike:.1f}×",
            ))

    return _deduplicate_events(events)


def _deduplicate_events(events: list[EmergenceEvent], min_gap: int = 10) -> list[EmergenceEvent]:
    """Remove overlapping events, keeping the most severe."""
    if not events:
        return events
    events.sort(key=lambda e: e.start_step)
    deduped = [events[0]]
    for e in events[1:]:
        if e.start_step - deduped[-1].end_step > min_gap:
            deduped.append(e)
        elif e.severity > deduped[-1].severity:
            deduped[-1] = e
    return deduped


def run_emergence_analysis(
    prices: np.ndarray,
    volumes: np.ndarray | None = None,
    spreads: np.ndarray | None = None,
) -> dict[str, list[EmergenceEvent]]:
    """Run all emergence detection analyses."""
    results = {}
    results["flash_crashes"] = detect_flash_crashes(prices)
    results["bubbles"] = detect_bubbles(prices)

    if spreads is not None:
        results["liquidity_crises"] = detect_liquidity_crisis(spreads)

    if volumes is not None:
        volatilities = np.abs(np.diff(np.log(prices + 1e-10)))
        if len(volumes) > len(volatilities):
            volumes = volumes[:len(volatilities)]
        results["herding"] = detect_herding(volumes, volatilities)

    return results
