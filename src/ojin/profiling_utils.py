"""Profiling utilities for FPS tracking and latency measurement."""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


class FPSTracker:
    """Track frames-per-second statistics for streaming workloads."""

    def __init__(self, id: str) -> None:  # noqa: D107
        self.id = id
        self.last_update_time = time.perf_counter() - 0.04
        self.total_frames = 0
        self.total_frames_time = 0
        self.partial_fps_history: list[float] = []
        self.fps_history: list[float] = []
        self.partial_frames = 0
        self.last_partial_time = time.perf_counter() - 0.04
        self.total_partial_time = 0
        self.is_running = False
        self.start()

    def start(self) -> None:
        """Mark the tracker as running and reset timing baselines."""
        # This method should be called when the first frame is generated
        # We are assuming first frame takes exactly 40ms for a proper average.
        # Oris actually takes around 1s to generate the first frame
        self.is_running = True
        self.last_update_time = time.perf_counter() - 0.04
        self.last_partial_time = time.perf_counter() - 0.04

    def stop(self) -> None:
        """Stop tracking and reset all counters."""
        self.partial_fps_history = []
        self.fps_history = []
        self.is_running = False
        self.partial_frames = 0
        self.total_partial_time = 0
        self.total_frames = 0
        self.total_frames_time = 0
        self.last_update_time = 0

    def update(self, num_frames: int) -> None:
        """Record newly produced frames and update timing."""
        self.total_frames += num_frames
        self.total_frames_time += time.perf_counter() - self.last_update_time
        self.last_update_time = time.perf_counter()

        self.partial_frames += num_frames
        self.total_partial_time += time.perf_counter() - self.last_partial_time
        self.last_partial_time = time.perf_counter()

    @property
    def average_fps(self) -> float:
        """Return the overall average FPS since the tracker started."""
        if self.total_frames <= 1:
            return 0

        return (self.total_frames) / (self.total_frames_time)

    def register_history(self) -> None:
        """Snapshot current average and partial FPS into the history lists."""
        self.fps_history.append(self.average_fps)
        self.partial_fps_history.append(self.partial_average_fps)

    def get_partial_fps_history(self) -> list[float]:
        """Return the recorded partial FPS history."""
        return self.partial_fps_history.copy()

    def get_fps_history(self) -> list[float]:
        """Return the recorded overall FPS history."""
        return self.fps_history.copy()

    @property
    def partial_average_fps(self) -> float:
        """Return the partial-window average FPS."""
        if self.partial_frames <= 1:
            return 0

        return (self.partial_frames) / (self.total_partial_time)

    def reset_partial_average(self) -> None:
        """Reset the partial-window counters."""
        self.partial_frames = 0
        self.total_partial_time = 0
        self.last_partial_time = time.perf_counter()

    def log(self) -> None:
        """Log current statistics."""
        logger.info(
            "%s : FPS=%.4f PartialFPS: %.4f total_frames:%d partial_frames:%d",
            self.id,
            self.average_fps,
            self.partial_average_fps,
            self.total_frames,
            self.partial_frames,
        )
        self.partial_frames = 0
        self.total_partial_time = 0
        self.last_partial_time = time.perf_counter()


@dataclass
class LatencyMeasure:
    """Single latency measurement with start/end timestamps."""

    id: str
    start_time: float = 0
    end_time: float = 0


class LatencyTracker:
    """Tracker that collects latency measurements by ID."""

    _measures: ClassVar[dict[str, list[LatencyMeasure]]] = {}
    _sample_counts: ClassVar[dict[str, int]] = {}
    _sample_totals: ClassVar[dict[str, float]] = {}
    _sample_max: ClassVar[dict[str, float]] = {}
    _sample_min: ClassVar[dict[str, float]] = {}

    @classmethod
    def reset(cls) -> None:
        """Clear accumulated latency state."""
        cls._measures.clear()
        cls._sample_counts.clear()
        cls._sample_totals.clear()
        cls._sample_max.clear()
        cls._sample_min.clear()

    @classmethod
    def _record_sample(cls, measure_id: str, duration: float) -> None:
        """Record a completed latency sample without retaining the measure object."""
        cls._sample_counts[measure_id] = cls._sample_counts.get(measure_id, 0) + 1
        cls._sample_totals[measure_id] = (
            cls._sample_totals.get(measure_id, 0.0) + duration
        )
        cls._sample_max[measure_id] = max(
            duration,
            cls._sample_max.get(measure_id, duration),
        )
        cls._sample_min[measure_id] = min(
            duration,
            cls._sample_min.get(measure_id, duration),
        )

    @classmethod
    def start_latency_measure(cls, measure_id: str) -> None:
        """Begin a new latency measurement for the given ID."""
        logger.info("Starting latency measure %s", measure_id)

        cls._measures[measure_id] = [
            m for m in cls._measures.get(measure_id, []) if m.end_time == 0
        ]

        if len(cls._measures[measure_id]) > 0:
            logger.warning(
                "Latency measure %s is already running, discarding older one",
                measure_id,
            )
            cls._measures[measure_id].pop()

        cls._measures[measure_id].append(
            LatencyMeasure(measure_id, time.perf_counter())
        )

    @classmethod
    def stop_latency_measure(cls, measure_id: str) -> None:
        """End the current latency measurement for the given ID."""
        measures = cls._measures.get(measure_id)
        if not measures:
            logger.warning("Latency measure %s was not running", measure_id)
            return

        last_measure = measures[-1]
        if last_measure.end_time == 0:
            last_measure.end_time = time.perf_counter()
            duration = last_measure.end_time - last_measure.start_time
            cls._record_sample(measure_id, duration)
            logger.info(
                "Stopping latency measure %s: %.4fs",
                measure_id,
                duration,
            )
        cls._measures[measure_id] = [m for m in measures if m.end_time == 0]

    @classmethod
    def _get_completed_measures(cls, measure_id: str) -> list[LatencyMeasure]:
        """Get only completed measures (those with end_time != 0)."""
        if measure_id not in cls._measures:
            return []
        return [m for m in cls._measures[measure_id] if m.end_time != 0]

    @classmethod
    def average(cls, measure_id: str) -> float:
        """Return the average latency for the given measure ID."""
        count = cls._sample_counts.get(measure_id, 0)
        if count == 0:
            return 0
        return cls._sample_totals[measure_id] / count

    @classmethod
    def max(cls, measure_id: str) -> float:
        """Return the maximum latency for the given measure ID."""
        if cls._sample_counts.get(measure_id, 0) == 0:
            return 0
        return cls._sample_max[measure_id]

    @classmethod
    def min(cls, measure_id: str) -> float:
        """Return the minimum latency for the given measure ID."""
        if cls._sample_counts.get(measure_id, 0) == 0:
            return 0
        return cls._sample_min[measure_id]

    @classmethod
    def log(cls) -> None:
        """Log statistics for all measures without clearing running measures."""
        for measure_id, count in cls._sample_counts.items():
            if count == 0:
                continue

            logger.info(
                "Latency %s NumMeasures: %d Avg: %.4fs Max: %.4fs Min: %.4fs",
                measure_id,
                count,
                cls.average(measure_id),
                cls.max(measure_id),
                cls.min(measure_id),
            )
