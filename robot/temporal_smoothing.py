"""
Majority-vote temporal smoothing over the raw per-frame predictions, so a
single noisy frame doesn't flip the robot's command.

    3 consecutive LEFT predictions -> LEFT confirmed -> move left
"""
from collections import Counter, deque


class TemporalSmoother:
    def __init__(self, window_size: int = 5, confirm_count: int = 3):
        """
        window_size: how many recent predictions to keep.
        confirm_count: how many of those (for the SAME class) are needed
                       before a gesture is "confirmed" and emitted.
        """
        if confirm_count > window_size:
            raise ValueError("confirm_count cannot exceed window_size")
        self.window = deque(maxlen=window_size)
        self.confirm_count = confirm_count
        self.last_confirmed = None

    def push(self, prediction: str):
        """Feed one new raw prediction, return a confirmed gesture or None."""
        self.window.append(prediction)
        counts = Counter(self.window)
        best_label, best_count = counts.most_common(1)[0]

        if best_count >= self.confirm_count and best_label != self.last_confirmed:
            self.last_confirmed = best_label
            return best_label
        return None

    def reset(self):
        self.window.clear()
        self.last_confirmed = None
