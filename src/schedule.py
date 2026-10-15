"""Which time points to keep ready around the one on screen.

This module must not import chimerax: the portable tests run it outside ChimeraX.

Time points are ranked by how soon the viewer is likely to need them. Frames ahead in the
direction of travel come first, frames behind cost twice as much (so a window holds about
two ahead for each one behind), and distances wrap around the series because the series
player loops. The same ranking says which frames to drop first when the budget is full.
"""

BEHIND_WEIGHT = 2.0
PLAYING_STREAK = 3


class Direction:
    """Direction of travel (+1 or -1) from the sequence of time points shown."""

    def __init__(self, nt, direction=1):
        self.nt = nt
        self.direction = direction
        self.last = None
        self.streak = 0  # consecutive moves in the same direction

    def update(self, t):
        """Note that time point t is now shown; returns the direction (+1 or -1).

        A move of up to a quarter of the series (taken round the loop) sets the direction;
        a bigger jump (the user dragged the slider) keeps the previous one.
        """
        if self.last is not None and t != self.last:
            d = offset(t, self.last, self.nt)
            if abs(d) <= max(1, self.nt // 4):
                new = 1 if d > 0 else -1
                self.streak = self.streak + 1 if new == self.direction else 1
                self.direction = new
            else:
                self.streak = 0
        self.last = t
        return self.direction

    @property
    def playing(self):
        """True after PLAYING_STREAK moves in a row the same way (playing or scrubbing)."""
        return self.streak >= PLAYING_STREAK


def offset(t, cur, nt):
    """Signed distance from cur to t going the short way round a series of nt points."""
    d = (t - cur) % nt
    return d - nt if d > nt // 2 else d


def score(t, cur, direction, nt):
    """How far t is from being needed: 0 for cur, smaller means sooner."""
    d = offset(t, cur, nt) * direction
    return float(d) if d >= 0 else -d * BEHIND_WEIGHT


def wanted(cur, direction, nt, capacity):
    """The time points to keep ready, most urgent first; at most `capacity` of them."""
    if nt <= 0 or capacity <= 0:
        return []
    ranked = sorted(range(nt), key=lambda t: (score(t, cur, direction, nt), -offset(t, cur, nt)
                                              * direction))
    return ranked[:min(capacity, nt)]
