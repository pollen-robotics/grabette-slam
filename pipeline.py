"""Episode → SLAM → LeRobot → Hub pipeline, run in-process inside the Space.

Thin glue over grabette_postprocess. The SLAM binary is bundled in the image
(no Docker), so slam.run_slam is called with binary=VSLAM_BINARY.
"""

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path

from grabette_postprocess.convert import convert_episode
from grabette_postprocess import slam
from grabette_postprocess.episode_manager import find_episodes
from grabette_postprocess.checks.trajectory import check_trajectory
from grabette_postprocess.checks.recording import check_recording
from grabette_postprocess.checks.sync import check_dcam_imu, check_image_trajectory

BINARY = os.environ.get("VSLAM_BINARY", "/usr/local/bin/offline_vslam")

# LeRobot's push_to_hub (v0.6.0) takes no token argument — it resolves the HF
# token from the process-global environment. Concurrent conversions run in
# separate threads sharing os.environ, so two overlapping pushes would clobber
# each other's token (wrong-account push / 401). Serialize the token-scoped push
# and hold the token in the env ONLY for its duration (see push_lerobot).
_PUSH_LOCK = threading.Lock()

# Optional dataset content the user can choose to keep (UI checkboxes, REST
# `keep`). Everything else (cam0, dcam, action, is_lost, depth_camera, tags)
# is always written. key -> UI label.
CONTENT_OPTIONS = {
    "depth": "Depth map (dcam_depth)",
    "right": "Right image (dcam_r, OAK-D only)",
    "imu": "IMU (OAK-D only)",
}


def content_kwargs(keep=None) -> dict:
    """Builder kwargs for the optional content kept by the user.

    keep: iterable of CONTENT_OPTIONS keys. Every listed stream is kept, and the
    episodes without it are skipped (a LeRobot schema is the same for every
    episode); an unlisted one is left out.
    keep=None (caller made no choice, e.g. an older fleet): the depth map is kept,
    and the right image / IMU only when every episode has them, so no episode is
    dropped for lacking them (see dataset._optional_stream_policy).
    """
    if keep is None:
        return {"with_depth": True, "with_right": None, "with_imu": None}
    keep = set(keep)
    unknown = keep - set(CONTENT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown content option(s): {', '.join(sorted(unknown))} "
                         f"(expected {', '.join(CONTENT_OPTIONS)})")
    return {
        "with_depth": "depth" in keep,
        "with_right": "right" in keep,
        "with_imu": "imu" in keep,
    }


# Log-line prefix of the dataset-content summary; views.content_recap lifts these
# lines into the end-of-run card.
CONTENT_LOG_PREFIX = "[content]"

# Feature name and why a stream may be missing, per optional stream.
_CONTENT_STREAMS = {
    "depth": ("dcam_depth", "depth map"),
    "right": ("dcam_r", "right image: OAK-D only"),
    "imu": ("imu", "IMU: OAK-D only"),
}


def content_plan(arm_dirs, keep=None, unit="episode"):
    """Decide the optional dataset content from what the episodes actually have,
    and explain it.

    arm_dirs: every per-arm episode dir that is about to be built.
    keep: the user's choice (CONTENT_OPTIONS keys), None = no choice (see below).
    unit: what one arm dir is called in the summary ("episode" / "arm recording").

    Returns (builder kwargs, summary lines). The kwargs are explicit True/False
    decisions, so the builder does exactly what the summary says:
      - ticked → kept; the episodes without it are skipped by the builder.
      - unticked → left out.
      - no choice (keep=None) → depth as ticked; right / IMU kept only when EVERY
        arm has them, otherwise left out with the count.
    Raises ValueError when a ticked stream is on no episode at all: every episode
    would be skipped, so there would be no dataset to build.
    """
    from grabette_postprocess.dataset import optional_streams_available
    from grabette_postprocess.episode_files import depth_camera_model

    wanted = content_kwargs(keep)
    arm_dirs = [Path(d) for d in arm_dirs]
    total = len(arm_dirs)
    avail = [optional_streams_available(d) for d in arm_dirs]
    count = {k: sum(a[k] for a in avail) for k in _CONTENT_STREAMS}
    units = f"{unit}(s)"

    decided = {}
    lines = ["always kept: cam0, dcam (left), action"]
    for key, (feature, why) in _CONTENT_STREAMS.items():
        choice = wanted[f"with_{key}"]  # True = ticked, False = unticked, None = auto
        n = count[key]
        if choice is False:
            decided[key] = False
            lines.append(f"{feature}: not kept — unticked (available on {n}/{total} {units})")
        elif choice is True:
            if total and n == 0:
                raise ValueError(f"{feature} is selected but no {unit} has it "
                                 f"({why}) — untick it to build the dataset.")
            decided[key] = True
            extra = (f" — {total - n}/{total} {units} without it will be skipped"
                     if n < total else "")
            lines.append(f"{feature}: kept{extra}")
        elif total and n == total:
            decided[key] = True
            lines.append(f"{feature}: kept ({n}/{total} {units})")
        else:
            decided[key] = False
            lines.append(f"{feature}: not kept — only on {n}/{total} {units} ({why})"
                         if n else f"{feature}: not kept — no {unit} has it ({why})")

    cameras: dict[str, int] = {}
    for d in arm_dirs:
        model = depth_camera_model(d)
        cameras[model] = cameras.get(model, 0) + 1
    lines.append("depth cameras: " + ", ".join(
        f"{m} ×{c}" for m, c in sorted(cameras.items())))

    kwargs = {"with_depth": decided["depth"], "with_right": decided["right"],
              "with_imu": decided["imu"]}
    return kwargs, lines


def content_gaps(arm_dirs, keep):
    """Ticked streams (in `keep`) that some of `arm_dirs` don't have.

    Returns [{"key", "label", "total", "missing": [(name, dir), …]}] — one entry
    per ticked stream with at least one arm dir lacking it. Empty when keep is
    None (no explicit choice: nothing is forced) or nothing is missing.
    """
    if keep is None:
        return []
    from grabette_postprocess.dataset import optional_streams_available

    arm_dirs = [Path(d) for d in arm_dirs]
    avail = {d: optional_streams_available(d) for d in arm_dirs}
    gaps = []
    for key, label in CONTENT_OPTIONS.items():
        if key not in keep:
            continue
        missing = [(episode_label(d), d) for d in arm_dirs if not avail[d][key]]
        if missing:
            gaps.append({"key": key, "label": label,
                         "total": len(arm_dirs), "missing": missing})
    return gaps


def apply_content_decisions(episodes, keep, gaps, decisions, log=print):
    """Apply the user's choice for each gap: "drop" removes the stream from keep,
    "exclude" removes the episodes without it. A stream no episode has can only be
    dropped. Returns (episodes, keep)."""
    keep = list(keep)
    excluded: set = set()
    for g in gaps:
        choice = "drop" if len(g["missing"]) == g["total"] else decisions.get(g["key"], "drop")
        if choice == "exclude":
            names = [n for n, _ in g["missing"]]
            excluded.update(d for _, d in g["missing"])
            log(f"  • {g['label']}: kept — excluding {len(names)} episode(s) without it: "
                f"{', '.join(names)}")
        else:
            keep.remove(g["key"])
            log(f"  • {g['label']}: not added ({len(g['missing'])}/{g['total']} "
                f"episode(s) don't have it)")
    return [ep for ep in episodes if Path(ep) not in excluded], keep


def _log_content(lines, log) -> None:
    log("Dataset content:")
    for ln in lines:
        log(f"  {CONTENT_LOG_PREFIX} {ln}")


def _check_sync(episode_dir):
    """Thin adapter: run check_dcam_imu and normalise to {verdict, message}."""
    result = check_dcam_imu(episode_dir)
    if result is None:
        return None
    lag_ms = abs(result["lag"]) * 1000
    note = result.get("note", "")
    msg = f"depth-cam ↔ gyro lag {lag_ms:.1f} ms"
    if note:
        msg += f" — {note}"
    return {"verdict": result["verdict"], "message": msg}


def find_episode_dirs(root: Path) -> list[Path]:
    """Every directory under root holding a raw depth-camera recording
    (dcam_left.mp4, or legacy oakd_left.mp4). Delegates to the package's single
    discovery helper."""
    # Use the default anchor: find_episodes searches the canonical dcam_ name AND
    # its legacy oakd_ counterpart. Passing the legacy name explicitly would
    # search ONLY that, making every new recording invisible here.
    return find_episodes(root)


def pair_episodes_by_role(arm_dirs) -> dict[str, dict[str, Path]]:
    """Group per-arm episode dirs into synchronized recordings for the bimanual
    build. The fleet uploads each device's stream to `{episode_id}/{role}/`, so
    an arm dir's PARENT name is the episode id and its OWN name is the role.
    Returns {episode_id: {role: arm_dir}}."""
    groups: dict[str, dict[str, Path]] = {}
    for d in arm_dirs:
        d = Path(d)
        groups.setdefault(d.parent.name, {})[d.name] = d
    return groups


# Device roles the fleet records with. A raw dataset uploaded by the fleet puts
# each device's stream in `{episode_id}/{role}/`, so an episode dir named after
# one of these is an ARM of a (possibly multi-arm) recording, not a recording.
# Episode ids are timestamps ("20250101_120000"), so there's no ambiguity.
ROLE_SLOTS = ("left", "right", "casquette")


def detect_roles(episode_dirs) -> list[str]:
    """The device roles present in a raw dataset, inferred from its layout.

    Returns the sorted roles when EVERY episode dir is a role dir (the fleet's
    `{episode_id}/{role}/` layout), else [] for the legacy flat layout uploaded
    by a lone grabette (`episodes/{episode_id}/`).

    This is what keeps the UI run consistent with the REST run: the API is told
    the roles explicitly by the fleet, while the UI only has the raw repo — so it
    reads them back off the layout instead of silently building a single-arm
    dataset from a bimanual recording."""
    names = {Path(d).name for d in episode_dirs}
    if names and names <= set(ROLE_SLOTS):
        return sorted(names)
    return []


def episode_label(ep: Path) -> str:
    """Unique, human-readable name for one episode dir.

    In the fleet's role layout the dir's own name is just the role ("left"), the
    same for every recording — so qualify it with the recording id. Used for log
    lines, quality entries and the review panel, where a name doubles as the key
    identifying which episode the user dropped."""
    ep = Path(ep)
    return f"{ep.parent.name}/{ep.name}" if ep.name in ROLE_SLOTS else ep.name


def build_recordings(target_repo, task, root, episodes, report_by_ep,
                     roles=None, source_user=None, log=print, keep=None) -> list:
    """Build the LeRobot dataset on disk from the kept episode dirs, picking the
    schema from the raw layout. Shared by the UI run (build_lerobot) and the
    headless REST run so both produce the SAME dataset for the same raw input.

    roles: the device roles present, when the caller already knows them (the REST
        API is told by the fleet). None/empty → inferred from the layout via
        detect_roles.

    With roles → ONE uniform, role-prefixed schema for any arm count (mono is the
    single-role case: left_cam0/left_dcam + 8D left action; bimanual adds the
    right block), built from the per-arm dirs paired back into synchronized
    recordings. Without → the legacy flat single-arm schema (cam0/dcam, 8D).

    keep: optional content to keep (CONTENT_OPTIONS keys, see content_kwargs);
        None = no choice (depth kept, right / IMU only when every episode has them).

    Returns the list of built recordings (recording ids in the role layout,
    episode dirs in the flat one); callers use its length as the episode count.
    """
    from grabette_postprocess.checks.tags import episode_tags
    from grabette_postprocess.dataset import build_dataset, build_dataset_multi

    content_kwargs(keep)  # validate before any work
    roles = sorted(roles) if roles else detect_roles(episodes)
    if not roles:
        tags_by_recording = {ep.name: episode_tags(ep, report_by_ep.get(ep))
                             for ep in episodes}
        content, lines = content_plan(episodes, keep, unit="episode")
        _log_content(lines, log)
        log(f"Building LeRobot dataset from {len(episodes)} episode(s)…")
        build_dataset(repo_id=target_repo, episode_dirs=episodes, task=task,
                      root=Path(root), source_user=source_user,
                      tags_by_recording=tags_by_recording, **content)
        return list(episodes)

    # Role layout: pair the per-arm dirs back into one LeRobot episode each. A
    # recording missing an arm can't be paired (the arm was dropped in review or
    # never uploaded) — skip it here rather than deep inside the builder, so the
    # returned count matches what is actually written.
    groups = pair_episodes_by_role(episodes)
    for ep_id in sorted(groups):
        missing = [r for r in roles if r not in groups[ep_id]]
        if missing:
            log(f"  • Skipping {ep_id}: missing arm(s) {', '.join(missing)}")
            groups.pop(ep_id)
    if not groups:
        raise RuntimeError(
            f"No recording has all the arms ({'+'.join(roles)}) — nothing to push.")
    # Tags keyed by recording id, prefixed by role so filtering stays attributable
    # per arm (e.g. left_tracking_lost).
    tags_by_recording = {
        ep_id: [f"{role}_{t}" for role in sorted(group)
                for t in episode_tags(group[role], report_by_ep.get(group[role]))]
        for ep_id, group in groups.items()
    }
    content, lines = content_plan(
        [group[r] for group in groups.values() for r in roles], keep,
        unit="arm recording")
    _log_content(lines, log)
    log(f"Building LeRobot dataset ({'+'.join(roles)}) from "
        f"{len(groups)} recording(s)…")
    build_dataset_multi(repo_id=target_repo, groups=groups, roles=roles, task=task,
                        root=Path(root), source_user=source_user,
                        tags_by_recording=tags_by_recording, **content)
    return sorted(groups)


def _run_parallel(episodes, fn, log, on_progress, phase, gate, should_stop):
    """Run fn(ep) -> (value, log_lines) over every episode concurrently, returning
    the list of values in completion order (callers sort/filter). Shared by the
    pre-SLAM check phase and the SLAM phase — both walk the same episode set with
    the same pause/stop semantics, only the per-episode work differs.

    Episodes are independent (each reads/writes only its own dir), so they run
    concurrently, capped at the core count (SLAM is a single-threaded CPU-bound
    subprocess; optical-flow likewise saturates one core).

    log_lines are collected per episode and emitted together rather than printed
    from the worker, so concurrent episodes don't interleave in the output.
    on_progress: optional callback(done, total, phase) fired per finished episode.
    gate: optional no-arg pause checkpoint, honored at dispatch time — before each
        episode is submitted, NOT inside the worker — so a paused run holds
        *between* episodes (it stops starting new ones) while any in-flight work
        finishes on its own (a SLAM subprocess can't be interrupted mid-call).
    should_stop: optional no-arg predicate; when True the loop stops collecting and
        cancels not-yet-started episodes.
    """
    total = len(episodes)
    if on_progress:
        on_progress(0, total, phase)
    workers = min(total, os.cpu_count() or 2)
    out = []
    done = 0
    pending = iter(episodes)

    def _dispatch_next():
        """Pause/stop checkpoint, then the next episode to start (or None when
        drained / stopped). gate() blocks here while paused, so no NEW episode is
        started until the user resumes — this is what makes Pause work mid-phase."""
        if gate:
            gate()  # blocks while paused; returns at once if a stop is requested
        if should_stop and should_stop():
            return None
        return next(pending, None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = set()
        for _ in range(workers):  # prime the pool (gated, so a pause holds at start)
            ep = _dispatch_next()
            if ep is None:
                break
            in_flight.add(pool.submit(fn, ep))

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                value, lines = fut.result()
                for line in lines:
                    log(line)
                out.append(value)
                done += 1
                if on_progress:
                    on_progress(done, total, phase)
            if should_stop and should_stop():
                log("⛔ Stop requested — abandoning remaining episodes")
                pool.shutdown(wait=False, cancel_futures=True)
                break
            while len(in_flight) < workers:  # refill (gated → Pause holds here)
                ep = _dispatch_next()
                if ep is None:
                    break
                in_flight.add(pool.submit(fn, ep))
    return out


def _precheck_episode(ep: Path, run_sync: bool = False):
    """Content + (optional) synchronization check for one raw episode, BEFORE SLAM
    (runs in a worker thread). Returns (result_dict, log_lines).

    result_dict = {ep, name, errors, warnings, sync} where sync is the
    depth-cam↔gyro verdict dict, or None when the sync check was disabled
    (run_sync=False, the default) or couldn't be computed. The caller uses
    errors/warnings/sync to decide which episodes to flag for the pre-SLAM review.
    Operates on the raw recording layout (before convert_episode).

    run_sync: the depth-cam↔gyro sync check is a debug-only diagnostic (opt-in);
    when False it's skipped entirely and sync stays None.
    """
    lines = []
    name = episode_label(ep)
    # require_right=False: the unused right camera stream shouldn't gate a run — its
    # absence is irrelevant to SLAM/dataset, even though we now download it.
    chk = check_recording(ep, require_right=False)
    if chk["errors"] or chk["warnings"]:
        verdict = "ERROR" if chk["errors"] else "WARN"
        lines.append(f"  [check/{verdict}] {name}")
        for msg in (*chk["errors"], *chk["warnings"]):
            lines.append(f"      • {msg}")

    sync = None
    if run_sync:
        try:
            sync = _check_sync(ep)
        except Exception as e:  # a sync failure must never abort the whole run
            lines.append(f"  ⚠️ {name}: sync check failed: {e}")
        if sync is not None:
            lines.append(f"  [sync/{sync['verdict']}] {name}: {sync['message']}")

    return {"ep": ep, "name": name, "errors": chk["errors"],
            "warnings": chk["warnings"], "sync": sync}, lines


def precheck_episodes(episode_dirs, log=print, should_stop=None, on_progress=None,
                      gate=None, run_sync=False) -> list[dict]:
    """Content check (check_recording) + optional arducam↔gyro sync on every
    episode, in parallel, BEFORE any SLAM. Returns the per-episode check dicts,
    sorted by name (see _precheck_episode for the dict shape).

    run_sync: when False (the default) the depth-cam↔gyro sync diagnostic is skipped;
    set True for debugging."""
    fn = partial(_precheck_episode, run_sync=run_sync)
    results = _run_parallel(episode_dirs, fn, log, on_progress,
                            "check", gate, should_stop)
    return sorted(results, key=lambda c: c["name"])


def _slam_episode(ep: Path, run_sync: bool = False):
    """Convert + SLAM + trajectory quality-check one episode (runs in a worker
    thread). Returns ((episode_dir | None, report | None), log_lines).

    The dir is None when SLAM produced no trajectory (the episode is dropped
    outright) — report is then None too. The trajectory verdict is returned so the
    caller can let the user drop flagged episodes before the dataset is built (see
    build_lerobot's `review` hook). Content/sync are NOT re-checked here — that
    happens in the pre-SLAM phase (precheck_episodes).

    run_sync: the post-SLAM image↔trajectory alignment is a debug-only diagnostic
    (opt-in, informational — it never drops an episode); when False it's skipped."""
    lines = []
    name = episode_label(ep)
    lines.append(f"▶ {name}: convert")
    convert_episode(ep)

    lines.append(f"▶ {name}: SLAM")
    r = slam.run_slam(ep, binary=BINARY, show_progress=False)
    if r.trajectory_path is None:
        lines.append(f"  ✗ {name}: SLAM failed (rc={r.returncode})")
        return (None, None), lines
    lines.append(f"  ✓ {name}: tracking {r.tracking_pct:.1f}% ({r.tracked_frames}/{r.total_frames})")

    report = check_trajectory(r.trajectory_path, ep / "slam_metadata.json")
    lines.append(
        f"  [{report.verdict}] {name}: dist={report.total_distance_m:.2f}m "
        f"med_step={report.median_step_mm:.1f}mm jumps={report.n_jumps}"
    )
    for msg in (*report.errors, *report.warnings):
        lines.append(f"      • {msg}")

    # Image↔pose temporal alignment (POST-SLAM): correlate the Arducam optical
    # flow against the trajectory's angular velocity — the end-to-end image↔pose
    # offset the policy trains on. Complements the pre-SLAM depth-cam↔gyro gate
    # (_check_sync), which only validates the SLAM inputs. Informational (never
    # drops); a sync failure must never abort the run. Debug-only (opt-in).
    if run_sync:
        try:
            sync = check_image_trajectory(ep)
        except Exception as e:
            sync = None
            lines.append(f"  ⚠️ {name}: image↔trajectory check failed: {e}")
        if sync is not None:
            msg = f"image↔trajectory lag {sync['lag'] * 1000:+.0f} ms (corr {sync['corr']:.2f})"
            if sync["note"]:
                msg += f" — {sync['note']}"
            lines.append(f"  [sync/{sync['verdict']}] {name}: {msg}")

    return (ep, report), lines


def run_slam(episode_dirs, log=print, should_stop=None, on_progress=None,
             gate=None, run_sync=False) -> list[tuple[Path, "TrajectoryReport"]]:
    """Convert + SLAM every given episode in parallel.

    Takes an explicit episode list (the pre-SLAM review may have dropped some) and
    returns a list of (episode_dir, trajectory_report) for every episode that
    produced a trajectory, sorted by episode name. SLAM-failed episodes are
    dropped outright. The report carries the quality verdict so the caller can
    offer the user a chance to drop flagged episodes before building.

    run_sync: forwarded to _slam_episode to enable the debug-only post-SLAM
    image↔trajectory alignment diagnostic (off by default)."""
    fn = partial(_slam_episode, run_sync=run_sync)
    results = _run_parallel(episode_dirs, fn, log, on_progress,
                            "slam", gate, should_stop)
    processed = [(ep, report) for ep, report in results if ep is not None]
    # Sort on the full path, not the dir name: in the fleet's {episode_id}/{role}
    # layout the name is just the role, so sorting by it would interleave the
    # recordings instead of keeping a recording's arms adjacent.
    return sorted(processed, key=lambda t: str(t[0]))


def _branch_name(task: str) -> str:
    """Deterministic branch name from the task — same input gives the same branch,
    so a retry pushes to the branch the first attempt was aiming for."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "update"
    return f"grabette-{slug}"


def _push_to_branch(ds, target_repo, branch, log, token, attempts=3):
    """Create the branch from main, then push_to_hub to it — retrying transient HF
    5xx errors, with an error message keyed on the real HTTP status (a 5xx is a
    server hiccup, not a token problem — only 401/403 mean the token can't write).

    We create the branch explicitly (from the repo's default branch) and pin
    ds.revision to it, because LeRobot's push_to_hub otherwise branches from its
    codebase version (e.g. "v3.0") — a ref that doesn't exist on a plain repo, so
    HF returns a *deterministic* 500.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for i in range(attempts):
        try:
            api.create_branch(target_repo, branch=branch, repo_type="dataset", exist_ok=True)
            ds.revision = branch  # makes push_to_hub's own create_branch a no-op
            ds.push_to_hub(branch=branch, tags=["lerobot", "grabette"], tag_version=False)
            return
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status and status >= 500 and i < attempts - 1:
                wait = 3 * (i + 1)
                log(f"  ⚠️ HF server error (HTTP {status}); retrying in {wait}s "
                    f"({i + 1}/{attempts - 1})…")
                time.sleep(wait)
                continue
            if status in (401, 403):
                raise RuntimeError(
                    f"Could not write to {target_repo} (HTTP {status}): your sign-in "
                    f"token appears read-only. Re-login to grant the Space the "
                    f"write-repos scope, then re-run."
                ) from e
            if status and status >= 500:
                raise RuntimeError(
                    f"Hugging Face had a server error (HTTP {status}) while pushing to "
                    f"branch '{branch}' on {target_repo}. The built dataset is cached — "
                    f"click “Retry push” to try again without re-running SLAM."
                ) from e
            raise RuntimeError(
                f"Could not push to branch '{branch}' on {target_repo}: {e}"
            ) from e


def build_lerobot(dataset_dir, target_repo, task, root, log=print,
                  should_stop=None, to_branch=False, on_progress=None,
                  token=None, gate=None, pre_review=None, review=None,
                  run_sync=False, keep=None, content_review=None) -> list:
    """Check + SLAM + build the LeRobot dataset on disk under `root`.

    Pipeline: completeness/sync prechecks → (pre_review drop) → convert+SLAM →
    (trajectory review drop) → build. Returns the list of built recordings — one
    entry per LeRobot episode written, so len() is the episode count to push (see
    build_recordings: recording ids in the fleet's role layout, episode dirs in
    the flat one). Does NOT push — call push_lerobot() afterwards. Splitting build from push lets
    a failed push be retried (the built dataset stays on disk) without re-running
    the slow SLAM step.

    should_stop: optional no-arg predicate; checked before the build so a stop
        request never produces a dataset.
    token: HF token (exported as HF_TOKEN; also used to resolve the username for
        the per-episode traceability sidecar when to_branch is set).
    gate: optional no-arg pause checkpoint; honored per episode during the check
        and SLAM phases and once more before the build step.
    pre_review: optional callback(checks) -> list[Path], called between the
        completeness/sync prechecks and SLAM with the list of per-episode check
        dicts. Returns the episode dirs to keep — letting the caller drop episodes
        flagged as incomplete or desynced BEFORE the slow SLAM runs. When None,
        every episode is kept.
    review: optional callback(results) -> list[Path], called between SLAM and the
        build with the list of (episode_dir, trajectory_report) tuples. It returns
        the episode dirs to keep — letting the caller drop episodes whose
        trajectory check came back flagged. When None, every episode is kept.
    run_sync: enable the debug-only sync diagnostics (pre-SLAM depth-cam↔gyro and
        post-SLAM image↔trajectory). Off by default.
    keep: optional content to keep in the dataset (CONTENT_OPTIONS keys, see
        content_kwargs); None = no choice.
    content_review: optional callback(gaps) -> {key: "drop" | "exclude"}, called
        after the trajectory review when ticked streams are missing from some of
        the kept episodes (see content_gaps). For each, the user decides: drop
        the stream, or exclude the episodes without it. Without it, a ticked
        stream excludes those episodes.
    """
    content_kwargs(keep)  # reject a bad option before the slow SLAM, not after
    # No global HF_TOKEN here: this function only builds locally and resolves the
    # username via an explicit HfApi(token=token).whoami() below; the actual push
    # (push_lerobot) scopes the token under a lock itself.
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (dcam_left.mp4 or legacy oakd_left.mp4) found under {dataset_dir}")
    log(f"Found {len(episodes)} episode(s) : "
        f"{', '.join(episode_label(ep) for ep in episodes)}")

    # ---- Pre-SLAM phase: completeness + sync checks, then optional drop --------
    checks = precheck_episodes(episodes, log=log, should_stop=should_stop,
                               on_progress=on_progress, gate=gate, run_sync=run_sync)
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if pre_review is not None:
        episodes = pre_review(checks)
    else:
        episodes = [c["ep"] for c in checks]
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not episodes:
        raise RuntimeError("All episodes were dropped in the pre-SLAM review — nothing to process.")
    if gate:
        gate()  # pause checkpoint before the (slow) SLAM phase

    # ---- SLAM phase ------------------------------------------------------------
    results = run_slam(episodes, log=log, should_stop=should_stop,
                       on_progress=on_progress, gate=gate, run_sync=run_sync)
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not results:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")

    # Episode review: let the caller drop flagged episodes before building. The
    # callback may block (e.g. waiting on user input) — keep it after the stop
    # check above so a cancel during SLAM never reaches it.
    if review is not None:
        processed = review(results)
    else:
        processed = [ep for ep, _ in results]
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not processed:
        raise RuntimeError("All episodes were dropped in review — nothing to push.")

    # Content review: a ticked stream some kept episodes lack → the user picks,
    # per stream, between not adding it and excluding those episodes.
    if content_review is not None:
        gaps = content_gaps(processed, keep)
        if gaps:
            decisions = content_review(gaps)
            if should_stop and should_stop():
                raise RuntimeError("Stopped by user — nothing pushed.")
            processed, keep = apply_content_decisions(processed, keep, gaps,
                                                      decisions or {}, log=log)
            if not processed:
                raise RuntimeError("Every episode was excluded for a missing "
                                   "feature — nothing to push.")
    if gate:
        gate()  # pause checkpoint before the (CPU-heavy) build

    # When pushing to a branch of an existing repo, tag each episode with the
    # source recording + user (meta/episode_sources.json) so episodes from
    # different users sharing one repo stay distinguishable.
    source_user = None
    if to_branch:
        try:
            from huggingface_hub import HfApi
            source_user = HfApi(token=token).whoami().get("name")
        except Exception as e:
            log(f"  ⚠️ couldn't resolve username for episode traceability: {e}")

    if on_progress:
        on_progress(0, 0, "build")
    # The schema follows the RAW layout (see build_recordings): a fleet upload
    # ({episode_id}/{role}) builds the role-prefixed schema with every arm it
    # contains — so a bimanual raw dataset yields a bimanual LeRobot dataset here
    # exactly as it does through the REST API. A flat legacy upload stays on the
    # original single-arm schema. Per-episode tags (fixed_gripper /
    # tracking_lost), auto-derived from the recording check + trajectory report,
    # are written into the LeRobot episode metadata by build_recordings so
    # kept-but-flagged episodes stay filterable downstream.
    report_by_ep = {ep: report for ep, report in results}
    return build_recordings(target_repo, task, root, processed, report_by_ep,
                            source_user=source_user, log=log, keep=keep)


def push_lerobot(target_repo, task, root, n_episodes, to_branch=False,
                 token=None, log=print, on_progress=None,
                 gate=None, private=False) -> tuple[int, str | None, str]:
    """Push an already-built LeRobot dataset (on disk under `root`) to the Hub.

    Separated from the build so a failed push can be retried without re-running
    SLAM. Loads the dataset from disk, so it works on a fresh process too.

    token: HF token used for the upload (also exported as HF_TOKEN, picked up by
        push_to_hub's internal HfApi).
    to_branch: if True, push to a dedicated branch instead of main (used when the
        target repo already exists, to leave main untouched). We push to a branch
        rather than opening a PR because HF "Sign in with HF" OAuth tokens can
        write content but are not allowed to open PRs (that needs the separate
        discussions/PR permission), so create_pr=True always 403s here.

    Returns (n_episodes, link_or_None, mode) where mode is "main" (link None) or
    "branch" (link = branch URL).
    """
    import os

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # push_to_hub reads the token from os.environ (no token arg). Serialize it and
    # set the token ONLY for the duration of the push, restoring the previous env
    # value after — so concurrent conversions never clobber each other's token and
    # none is left lingering in the process environment. The heavy phases
    # (download, SLAM, build) run outside this lock and stay parallel.
    with _PUSH_LOCK:
        _prev_token = os.environ.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
        try:
            ds = LeRobotDataset(target_repo, root=Path(root))

            if gate:
                gate()  # pause checkpoint before the upload
            if on_progress:
                on_progress(0, 0, "push")
            if to_branch:
                branch = _branch_name(task)
                log(f"Pushing to branch '{branch}' on {target_repo} …")
                _push_to_branch(ds, target_repo, branch, log, token)
                branch_url = f"https://huggingface.co/datasets/{target_repo}/tree/{branch}"
                log(f"✅ Pushed to branch '{branch}'.")
                return n_episodes, branch_url, "branch"

            log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
            try:
                ds.push_to_hub(tags=["lerobot", "grabette"], private=private)
            except Exception as e:
                # This path creates the repo (target didn't exist at pre-flight). A
                # 401/403 here means the sign-in token lacks the 'manage-repos' scope
                # creation needs — pre-flight normally catches this, so reaching it
                # implies partial/stale consent. Give an actionable message.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    raise RuntimeError(
                        f"Could not create '{target_repo}' (HTTP {status}): creating a new "
                        f"dataset needs the 'manage-repos' permission, which your sign-in "
                        f"token is missing. Sign out and back in to grant it, then re-run."
                    ) from e
                raise
            return n_episodes, None, "main"
        finally:
            # Restore the previous env token (or clear it) so nothing leaks out.
            if _prev_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = _prev_token
