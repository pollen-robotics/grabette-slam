"""Per-session run state — one RunSession per browser session, never shared.

Several people use the Space at the same time, each in their own Gradio session.
Nothing about a run may leak from one session into another: not the cooperative
control flags (pause / cancel), and above all not the flagged episodes published
by the check / sync / trajectory review gates — those name someone else's dataset.

So every flag that used to be a module-level global lives here, on a RunSession
looked up by the caller's `gr.Request.session_hash`. Handlers annotate a
`request: gr.Request` parameter (Gradio injects it — it is NOT a UI input) and
call `session.get(request)`; the worker thread captures the object itself, so a
run keeps working on its own flags even after its page is gone.
"""

import threading
import time

import gradio as gr

# A session is dropped once its page has been silent for this long. The review
# gr.Timer ticks twice a second while a page is open, so anything idle for hours
# is a closed tab (Gradio's unload event isn't relied on — a killed tab or a lost
# connection never fires it).
_TTL_S = 6 * 3600

_sessions: dict[str, "RunSession"] = {}
_lock = threading.Lock()


class RunSession:
    """One user's run: cooperative flags + the review gate's published state.

    stop           — abandon the run (Cancel confirmed)
    pause          — hold the worker at the next safe checkpoint (Pause)
    cancel_pending — Cancel was clicked, awaiting confirmation
    review*        — the gate the worker blocks on while the user triages the
                     flagged episodes (see review.py)
    """

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.cancel_pending = threading.Event()
        # A run has three sequential review gates — pre-SLAM (kind "input":
        # completeness + sync), post-SLAM (kind "trajectory") and, before the
        # build, missing features (kind "content") — so the same fields are
        # reused; `review_kind` tells the panel which gate it is drawing.
        self.review = threading.Event()       # worker is awaiting the user's decision
        self.review_done = threading.Event()  # Continue clicked → release the worker
        self.review_items: list[dict] = []    # flagged episodes published for the panel
        self.review_drop: list[str] = []      # episode names the user chose to drop
        self.review_choices: dict[str, str] = {}  # content gate: feature → drop/exclude
        self.review_kind = "trajectory"
        self.seen = time.monotonic()

    # ---- review gate ----------------------------------------------------
    def review_state(self) -> dict:
        """Snapshot for the reactive @gr.render; `open` drives whether it shows."""
        return {"open": self.review.is_set(),
                "kind": self.review_kind,
                "items": list(self.review_items),
                "dropped": list(self.review_drop),
                "choices": dict(self.review_choices)}

    def reset_review(self) -> None:
        """Clear every review flag (called by the controller's run reset)."""
        self.review.clear()
        self.review_done.clear()
        self.review_drop.clear()
        self.review_choices.clear()
        self.review_kind = "trajectory"

    # ---- control flags --------------------------------------------------
    def start_run(self) -> None:
        """Fresh run: clear every flag left over from the previous one."""
        self.stop.clear()
        self.pause.clear()
        self.cancel_pending.clear()
        self.reset_review()

    def wait_if_paused(self) -> None:
        """Block while paused (the worker's checkpoint gate). Returns at once if a
        stop/cancel is requested, so pausing then cancelling never deadlocks."""
        while self.pause.is_set() and not self.stop.is_set():
            time.sleep(0.2)


def get(request: gr.Request | None) -> RunSession:
    """The RunSession for this browser session, created on first use.

    `request` is None only if Gradio couldn't inject one (it always can for event
    handlers); a shared fallback session is used then rather than crashing — it is
    the one case where two callers could meet, and it can't be reached from the UI.
    """
    key = getattr(request, "session_hash", None) or "_fallback"
    now = time.monotonic()
    with _lock:
        for stale in [k for k, s in _sessions.items() if now - s.seen > _TTL_S]:
            del _sessions[stale]
        sess = _sessions.get(key)
        if sess is None:
            sess = _sessions[key] = RunSession()
        sess.seen = now
        return sess
