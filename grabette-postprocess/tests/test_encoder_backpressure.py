"""Tests for the streaming-encoder back-pressure hotfix
(grabette_postprocess.dataset._wait_for_encoder_room). Remove with the helper
when bumping to lerobot 0.7.1 (huggingface/lerobot#4462)."""
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from grabette_postprocess.dataset import _wait_for_encoder_room


def _dataset_with_queues(*queues):
    enc = SimpleNamespace(_frame_queues={f"cam{i}": q for i, q in enumerate(queues)})
    return SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=enc))


def test_no_streaming_encoder_is_a_noop():
    _wait_for_encoder_room(SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=None)))
    _wait_for_encoder_room(SimpleNamespace())  # other internals: no attributes at all


def test_returns_immediately_when_queues_have_room():
    q = queue.Queue(maxsize=2)
    q.put(0)
    t0 = time.monotonic()
    _wait_for_encoder_room(_dataset_with_queues(q, queue.Queue(maxsize=2)))
    assert time.monotonic() - t0 < 0.1


def test_waits_until_a_full_queue_is_drained():
    full = queue.Queue(maxsize=1)
    full.put(0)
    drainer = threading.Timer(0.2, full.get)
    drainer.start()
    t0 = time.monotonic()
    _wait_for_encoder_room(_dataset_with_queues(queue.Queue(maxsize=1), full))
    assert time.monotonic() - t0 >= 0.15
    assert full.qsize() == 0
    drainer.join()


def test_raises_when_the_encoder_makes_no_progress():
    stuck = queue.Queue(maxsize=1)
    stuck.put(0)
    with pytest.raises(RuntimeError, match="no progress"):
        _wait_for_encoder_room(_dataset_with_queues(stuck), timeout_s=0.05)
