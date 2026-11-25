from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from ..constants import REPEAT_WINDOW


class MachineRepeatIndex:
    """Causal rolling index: has the same machine emitted this exact value before?

    One state per sensor. ``transform`` must be called in chronological order
    when the same index is reused across slices (train -> validation -> test,
    or production batches): a row only sees the past, never the future.

    A frozen sensor repeats its own value; it does not need another machine to
    emit the same value. This index therefore counts only the current machine's
    previous readings, never the rest of the fleet.
    """

    def __init__(self, sensors: list[str], window: pd.Timedelta = REPEAT_WINDOW):
        self.sensors = list(sensors)
        self.window = pd.Timedelta(window)
        self._events = {sensor: deque() for sensor in self.sensors}
        self._counts = {sensor: defaultdict(Counter) for sensor in self.sensors}

    def _purge(self, sensor: str, now) -> None:
        """Drop events older than the window, updating the per-machine counts."""
        cutoff = now - self.window
        events = self._events[sensor]
        counts = self._counts[sensor]
        while events and events[0][0] < cutoff:
            _, machine, value = events.popleft()
            counter = counts[machine]
            counter[value] -= 1
            if counter[value] == 0:
                del counter[value]
            if not counter:
                del counts[machine]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``n_previous_same_machine_<sensor>`` for every row of ``frame``.

        The count is the number of previous readings of the *same machine* that
        carried the same value inside the window. The current row is not
        counted, so a value must have been seen on an earlier scan to count.
        """
        # ``order`` maps a sorted position back to the row's position in
        # ``frame``. Counts are written in the original order, so an input with
        # a duplicated index is returned aligned row by row instead of failing.
        order = frame["timestamp"].argsort(kind="stable").to_numpy()
        ordered = frame.iloc[order]
        n_rows = len(ordered)
        counts = {sensor: np.zeros(n_rows, dtype=np.int32) for sensor in self.sensors}
        timestamps = ordered["timestamp"].to_numpy()
        machines = ordered["machine_id"].to_numpy()
        values = {sensor: ordered[sensor].to_numpy() for sensor in self.sensors}

        start = 0
        while start < n_rows:
            end = start + 1
            timestamp = timestamps[start]
            while end < n_rows and timestamps[end] == timestamp:
                end += 1

            for sensor in self.sensors:
                self._purge(sensor, timestamp)

            # Count the past before adding the current snapshot.
            for position in range(start, end):
                row = order[position]
                machine = machines[position]
                for sensor in self.sensors:
                    value = values[sensor][position]
                    if pd.notna(value):
                        counts[sensor][row] = self._counts[sensor][machine][float(value)]

            for position in range(start, end):
                machine = machines[position]
                for sensor in self.sensors:
                    value = values[sensor][position]
                    if pd.notna(value):
                        self._events[sensor].append((timestamp, machine, float(value)))
                        self._counts[sensor][machine][float(value)] += 1

            start = end

        return pd.DataFrame(
            {f"n_previous_same_machine_{sensor}": counts[sensor] for sensor in self.sensors},
            index=frame.index,
        )
