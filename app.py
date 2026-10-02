"""Grabette SLAM → LeRobot pipeline — HuggingFace Space (Gradio + HF OAuth).

The user signs in with their HF account; the OAuth token is used to download the
source dataset and push the generated LeRobot dataset under their account.
Gradio auto-injects the gr.OAuthToken parameter — it is NOT a UI input.

This module is just the UI: it builds the Blocks layout and wires events. The
rendering helpers live in views.py, the streaming run generators in controller.py,
the episode-review panel in review.py, and every piece of run state in session.py —
per browser session, so concurrent users never see each other's run or its flagged
episodes.

REST API (called by grabette devices):
  POST /api/process  {source_repo, target_repo, task}  Authorization: Bearer <hf_token>  → {job_id}
  GET  /api/status/{job_id}                                      → {status, log, error}
  GET  /api/quality/{target_repo}   (no auth, like /api/status)   → {status, quality, flagged, …}

A "SLAM check" is the same pipeline run with check_only: it reports per-episode
tracking quality and pushes a dataset ONLY when an episode is flagged, so the
operator can look at the bad take in the LeRobot visualizer.
"""

import contextlib
import os
import tempfile
import threading
import uuid
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel

import review
from views import CSS, THEME, bar, space_banner, visualizer_url
from controller import (
    confirm_cancel,
    keep_running,
    request_cancel,
    reset,
    retry_push,
    run_pipeline,
    run_pipeline_branch,
    toggle_pause,
)

# ── Which deployment this process IS. HF sets SPACE_ID ("owner/grabette-slam-test")
# in every Space; absent when run locally, which reads as prod (see space_banner).
SPACE_ID = os.environ.get("SPACE_ID", "")
# Which ref the deploy workflow last pushed HERE. Set as Space *variables* by
# .github/workflows/deploy-branch-to-hf-test-space.yml rather than committed as a
# stamp file: the deployed tree then stays byte-identical to the branch it came
# from, and the Dockerfile needs no COPY for a file that only exists on the test
# Space. The price is that pushing to a Space by hand leaves the previous run's
# stamp in place — so the banner words it as the last *deploy*, and an unset
# variable is reported as unknown rather than quietly omitted.
DEPLOYED_REF = os.environ.get("GRABETTE_DEPLOYED_REF", "")
DEPLOYED_SHA = os.environ.get("GRABETTE_DEPLOYED_SHA", "")

# ── Standalone FastAPI app — Gradio is mounted onto it so custom routes survive.
# Registering routes on demo.app before launch() is unreliable in Gradio 6: the
# internal FastAPI instance may be replaced, dropping the handlers.
fastapi_app = FastAPI()

with gr.Blocks(title="Grabette SLAM → LeRobot") as demo:
    # Above the header, and only on a test deployment — production renders no
    # component at all rather than an empty div taking up a margin.
    if _BANNER := space_banner(SPACE_ID, DEPLOYED_REF, DEPLOYED_SHA):
        gr.HTML(_BANNER)
    gr.HTML(
        "<div id='app-header'><h1>Grabette post-processing</h1>"
        "<p>Raw Grabette recording → SLAM → LeRobot dataset on the Hub</p></div>"
    )
    with gr.Row():
        # ---- Inputs (left) ----
        with gr.Column(scale=2, min_width=280):
            gr.LoginButton(size="sm")
            source = gr.Textbox(label="Source dataset", info="raw Grabette repo on the Hub",
                                placeholder="pollen-robotics/grabette-raw")
            target = gr.Textbox(label="Target dataset", info="repo to create or contribute to",
                                placeholder="your-account/grabette-dataset")
            task = gr.Textbox(label="Task description", placeholder="cup grasping")
            # Sync checks (OAK cam↔gyro pre-SLAM, image↔trajectory post-SLAM) are a
            # debug-only diagnostic — off by default so a normal run skips them.
            debug_sync = gr.Checkbox(
                label="Run sync checks (debug)", value=False,
                info="Extra camera↔IMU / image↔trajectory timing diagnostics")
            export_tactile = gr.Checkbox(
                label="Export tactile data", value=True,
                info="Add the tactile sensor streams to the dataset")
            # Read-only recap of the inputs, shown in their place while running.
            selection_card = gr.HTML(visible=False)
            # Idle: only Run. Running: Run greyed to 'Running', + Pause + Cancel.
            # Finished: only Reset. All visibility is driven by the generators.
            with gr.Row():
                run_btn = gr.Button("▶  Run", variant="primary", scale=2)
                pause_btn = gr.Button("⏸  Pause", variant="secondary",
                                      scale=1, visible=False)
                cancel_btn = gr.Button("✕  Cancel", variant="stop",
                                       scale=1, visible=False)
            # Cancel confirmation (revealed by Cancel; replaces Pause/Cancel).
            with gr.Row():
                confirm_cancel_btn = gr.Button("✓  Confirm cancel", variant="stop",
                                               size="sm", visible=False)
                keep_btn = gr.Button("Keep running", variant="secondary",
                                     size="sm", visible=False)
            # Revealed (as the primary action, replacing Run) only when the target
            # dataset already exists (see the controller).
            confirm_branch_btn = gr.Button("⎇  Push to a new branch",
                                           variant="primary", visible=False)
            # Revealed only when a build succeeded but the push failed.
            retry_btn = gr.Button("⟳  Retry push", variant="secondary",
                                  size="sm", visible=False)
            # Shown once a run reaches a terminal state.
            reset_btn = gr.Button("↺  Reset", variant="secondary", visible=False)
            # Holds the built-but-unpushed dataset context for retry_push.
            retry_state = gr.State(None)
        # ---- Outputs (right) ----
        with gr.Column(scale=3, min_width=280):
            # Bar + log are revealed only while a run is in progress (running /
            # paused); the generators toggle their visibility via views.io().
            bar_out = gr.HTML(bar(0, "ready"), visible=False)
            log_out = gr.Textbox(label="Log", lines=15, max_lines=15, autoscroll=True,
                                 elem_id="logbox", visible=False)

            # Episode review (reactive): a gr.Timer mirrors THIS session's worker
            # review flags into review_state, and review.build_panel draws one orange
            # card per still-kept flagged episode. Not part of the streaming outputs.
            # gr.State is per-session and review.poll reads the caller's own
            # session.RunSession, so a flagged episode never reaches another user.
            review_state = gr.State({"open": False, "kind": "trajectory",
                                     "items": [], "dropped": []})
            review_timer = gr.Timer(0.5)
            review.build_panel(review_state)

            # line_breaks=True so a single "\n" in a summary/warning renders as a
            # line break (GFM soft breaks) — plain Markdown would collapse it to a
            # space and only "\n\n" (a full paragraph gap) would show.
            summary_out = gr.Markdown(line_breaks=True)

    # Order MUST match the generators' yields (views.io → bar, log; then summary,
    # retry_state; then views.btns → branch, retry, run, pause, cancel,
    # confirm_cancel, keep, reset; then views.inputs_view → source, target, task,
    # selection_card). The review panel is reactive, NOT in this list.
    outputs = [bar_out, log_out, summary_out, retry_state,
               confirm_branch_btn, retry_btn,
               run_btn, pause_btn, cancel_btn, confirm_cancel_btn, keep_btn, reset_btn,
               source, target, task, selection_card]
    run_event = run_btn.click(run_pipeline,
                              inputs=[source, target, task, debug_sync, export_tactile],
                              outputs=outputs)
    branch_event = confirm_branch_btn.click(run_pipeline_branch,
                                            inputs=[source, target, task, debug_sync,
                                                    export_tactile],
                                            outputs=outputs)
    retry_event = retry_btn.click(retry_push, inputs=[retry_state], outputs=outputs)
    reset_btn.click(reset, inputs=None, outputs=outputs)
    # Pause/Resume + cancel-confirm just flip cooperative flags; the running
    # generator re-renders the buttons from them on its next tick (outputs=None
    # → these never write the same components as the stream, so no conflict).
    pause_btn.click(toggle_pause, inputs=None, outputs=None)
    cancel_btn.click(request_cancel, inputs=None, outputs=None)
    keep_btn.click(keep_running, inputs=None, outputs=None)
    # Mirror this session's review flags into review_state every tick; review.poll
    # returns gr.skip() when nothing changed, so the @gr.render block only redraws
    # on a real open/close or drop-set change (no churn while idle or running).
    review_timer.tick(review.poll, inputs=review_state, outputs=review_state)
    # Confirm cancel: first cancel the streaming generator + signal the worker to
    # abandon (outputs=None so it preempts at once), THEN reset the page to idle.
    # Split into click(cancel) → .then(reset) because a single function can't both
    # cancel a generator and reliably write that generator's outputs.
    cancel_evt = confirm_cancel_btn.click(
        confirm_cancel, inputs=None, outputs=None,
        cancels=[run_event, branch_event, retry_event])
    cancel_evt.then(reset, inputs=None, outputs=outputs)


# ── REST API for grabette devices ────────────────────────────────────────────
# Headless pipeline calls: grabette uploads a raw dataset then POSTs here.
# No review callbacks — unattended run, bad episodes are skipped automatically.

_rest_jobs: dict[str, dict] = {}  # job_id → {status, log, error, quality, target_repo, …}

# A target repo whose NAME carries one of these is a fleet trajectory check, not a
# dataset the operator wants to keep: run the precheck + SLAM for the report, and
# push only if something came out flagged (see check_only below). Kept in sync with
# the grabette-fleet Space, which is what builds the name.
#
# Two of them, and both stay: the fleet names a check "{task}_trajectorycheck_{stamp}"
# now and "slamcheck-…" before, so a fleet that hasn't been redeployed yet is still
# sending the old one — accepting both is what lets the two Spaces be deployed in
# either order. The marker is delimited by underscores rather than matched bare, so
# an ordinary dataset that merely mentions a trajectory check isn't mistaken for one.
SLAM_CHECK_MARKER = "_trajectorycheck_"
SLAM_CHECK_PREFIX = "slamcheck-"   # legacy, still honoured


class _ProcessReq(BaseModel):
    source_repo: str
    target_repo: str
    task: str
    exclude_fail: bool = False
    exclude_bad: bool = False
    private: bool = False
    run_sync: bool = False
    tactile: bool = True
    # Device roles present in the raw dataset (e.g. ["left","right"]). >1 role
    # → bimanual build (each episode combines the arms, uploaded by the fleet to
    # {episode_id}/{role}/); ≤1 role (or empty) → the classic single-arm build.
    roles: list[str] = []
    # Leave the raw source dataset in place after a successful conversion. Defaults
    # to True: the raw is currently NEVER auto-deleted — re-uploading it from the
    # devices is the slow part of a rebuild, and a caller that cancels mid-build
    # would otherwise be left with a half-deleted source. The fleet sends this
    # explicitly (see GRABETTE_KEEP_RAW_DATASET there); pass False to restore the
    # delete-on-success behaviour.
    keep_raw: bool = True
    # SLAM check: run the precheck + SLAM to REPORT quality, and build/push the
    # dataset only if at least one episode is flagged (verdict != GOOD) — a clean
    # check leaves nothing on the Hub. Sent by the fleet and forwarded by the device
    # relay; a caller too old to send it still gets the same behaviour by naming the
    # target repo with SLAM_CHECK_MARKER (or the legacy prefix).
    check_only: bool = False


def _run_pipeline_headless(job_id: str, source_repo: str, target_repo: str,
                            task: str, token: str,
                            exclude_fail: bool = False,
                            exclude_bad: bool = False,
                            private: bool = False,
                            run_sync: bool = False,
                            tactile: bool = True,
                            roles: list[str] | None = None,
                            keep_raw: bool = True,
                            check_only: bool = False) -> None:
    """Run the pipeline synchronously in a background thread.

    exclude_fail: drop episodes with trajectory verdict FAIL.
    exclude_bad:  drop all BAD episodes.
    private:      create the LeRobot dataset repo as private.
    run_sync:     enable the debug-only sync diagnostics (off by default).
    tactile:      export the tactile streams into the dataset (on by default).
    keep_raw:     leave the raw source dataset alone (the default — nothing
                  auto-deletes a raw for now).
    check_only:   SLAM check — report quality, and push the dataset only if an
                  episode is flagged (see the push decision below).
    All episodes appear in the returned quality data (including excluded ones)
    so the caller can display a full recap.
    """
    logs: list[str] = []
    quality_episodes: list[dict] = []
    # Publish the (still growing) per-episode list on the job record right away:
    # it is the SAME list object, so /api/status and /api/quality expose verdicts
    # as SLAM works through the set instead of only once the run ends. That live
    # view is what lets a caller fill a report table episode by episode.
    _rest_jobs[job_id]["quality"] = quality_episodes

    def log(msg: str) -> None:
        logs.append(msg)
        _rest_jobs[job_id]["log"] = "\n".join(logs)

    def flagged() -> list[str]:
        """The episodes whose SLAM outcome is not clean: a BAD/FAIL trajectory
        (what "the take lost tracking" looks like here), a SLAM failure, or a
        pre-check ERROR that stopped the take from being run at all.

        A pre-check WARNING is deliberately NOT one. It is still reported — the
        caller shows it against the episode — but it says nothing about tracking,
        and "the gripper barely moved in this take" would otherwise push a dataset
        on nearly every check, which teaches the operator to ignore the result."""
        return [q["name"] for q in quality_episodes
                if q.get("verdict") != "GOOD"
                and not (q.get("kind") == "pre_check" and q.get("verdict") == "WARN")]

    result_link: str | None = None
    try:
        # No global os.environ["HF_TOKEN"] here — every HF call below passes the
        # token explicitly (HfApi / snapshot_download / delete_repo), and the push
        # scopes it under a lock itself (see push_lerobot). Setting it globally
        # would let concurrent conversions clobber each other's token.
        api = HfApi(token=token)
        log(f"Downloading {source_repo}…")
        work = Path(tempfile.mkdtemp())
        raw = snapshot_download(source_repo, repo_type="dataset",
                                local_dir=work / "raw", token=token,
                                max_workers=48)
        log("Download complete. Running SLAM + LeRobot conversion…")
        ds_root = work / "lerobot"

        with contextlib.redirect_stdout(type("W", (), {"write": lambda s, m: log(m.rstrip()) or None, "flush": lambda s: None})()): # noqa: E501
            from pipeline import (build_recordings, episode_label,
                                  find_episode_dirs, precheck_episodes, run_slam,
                                  push_lerobot)

            episodes = find_episode_dirs(Path(raw))
            if not episodes:
                raise ValueError(f"No episodes found under {raw}")
            log(f"Found {len(episodes)} episode(s).")

            # Pre-check: record quality data for all episodes; exclude those with
            # recording errors unconditionally, and optionally exclude those with
            # recording warnings or poor sync depending on caller flags.
            checks = precheck_episodes(episodes, log=log, run_sync=run_sync)
            for c in checks:
                sync = c.get("sync") or {}
                sync_verdict = sync.get("verdict", "GOOD")
                sync_msg = sync.get("message", "")
                sync_flagged = sync_verdict != "GOOD"
                if c["errors"]:
                    quality_episodes.append({
                        "name": c["name"], "kind": "pre_check", "verdict": "ERROR",
                        "errors": list(c["errors"]),
                        "warnings": list(c["warnings"]) + ([sync_msg] if sync_flagged and sync_msg else []),
                        "excluded": True,
                    })
                elif c["warnings"] or sync_flagged:
                    quality_episodes.append({
                        "name": c["name"], "kind": "pre_check", "verdict": "WARN",
                        "errors": [],
                        "warnings": list(c["warnings"]) + ([sync_msg] if sync_flagged and sync_msg else []),
                        "excluded": False,
                    })
            slam_input = [c["ep"] for c in checks if not c["errors"]]
            if not slam_input:
                log("⚠ All episodes failed the recording check — nothing to process.")
            else:
                # SLAM: convert + trajectory per episode.
                slam_results = run_slam(slam_input, log=log, run_sync=run_sync)
                slam_succeeded = {ep for ep, _ in slam_results}

                # Record SLAM failures (no trajectory produced at all).
                for ep in slam_input:
                    if ep not in slam_succeeded:
                        quality_episodes.append({
                            "name": episode_label(ep), "kind": "slam", "verdict": "FAIL",
                            "tracking_pct": 0.0, "n_jumps": 0, "total_distance_m": 0.0,
                            "errors": ["SLAM failed: no trajectory produced"],
                            "warnings": [], "excluded": True,
                        })

                # Apply verdict filter and record trajectory quality.
                kept = []
                for ep, report in slam_results:
                    excluded = (
                        (exclude_fail and report.verdict == "FAIL")
                        or (exclude_bad and report.verdict == "BAD")
                    )
                    if excluded:
                        log(f"  • Excluded ({report.verdict}): {episode_label(ep)}")
                    else:
                        kept.append(ep)
                    quality_episodes.append({
                        "name": episode_label(ep), "kind": "trajectory",
                        "verdict": report.verdict,
                        "tracking_pct": round(report.tracking_pct, 1),
                        # Frame counts, not just the percentage: "3 lost frames out
                        # of 900" and "90 out of 900" both round to a reassuring
                        # tracking_pct, and only the raw count says which is which.
                        "n_frames": report.n_frames,
                        "n_tracked": report.n_tracked,
                        "n_lost": report.n_lost,
                        "n_jumps": report.n_jumps,
                        "total_distance_m": round(report.total_distance_m, 2),
                        "errors": list(report.errors),
                        "warnings": list(report.warnings),
                        "excluded": excluded,
                    })

                bad = flagged()
                if check_only and not bad:
                    # A clean SLAM check leaves nothing behind: the answer IS the
                    # report ("nobody lost tracking, keep recording the same way"),
                    # and a dataset nobody asked for would just pile up on the Hub.
                    log(f"✅ SLAM check: {len(kept)} episode(s), none lost tracking "
                        "— nothing pushed.")
                elif not kept:
                    log("⚠ No episode produced an acceptable trajectory — dataset not pushed.")
                else:
                    if check_only:
                        # Flagged → push the WHOLE tested set, not just the culprits,
                        # so the bad take can be compared in the LeRobot visualizer
                        # against the ones that came out fine.
                        log(f"SLAM check flagged {len(bad)} episode(s) ({', '.join(bad)}) "
                            f"— pushing the {len(kept)} tested episode(s) for inspection.")
                    # Same builder as the UI run: `roles` (given by the fleet, raw
                    # uploaded to {episode_id}/{role}) selects the uniform
                    # role-prefixed schema for any arm count; when absent it is
                    # inferred from the layout, so a legacy flat upload keeps the
                    # original single-arm schema.
                    report_by_ep = {ep: rep for ep, rep in slam_results}
                    built = build_recordings(target_repo, task, ds_root, kept,
                                             report_by_ep, roles=roles, log=log,
                                             tactile=tactile)
                    n_built = len(built)
                    n, link, _mode = push_lerobot(target_repo, task, ds_root,
                                                  n_built, token=token, log=log,
                                                  private=private)
                    result_link = link or f"https://huggingface.co/datasets/{target_repo}"
                    log(f"✅ Done — {n} episode(s).")

        _rest_jobs[job_id].update(
            status="done",
            result=result_link,
            quality=quality_episodes,
            flagged=flagged(),
            # The LeRobot visualizer can only open a PUBLIC dataset, so a private
            # push gets no visualizer link rather than one that dead-ends.
            visualizer=visualizer_url(target_repo) if (result_link and not private) else None,
        )
    except Exception as e:
        _rest_jobs[job_id].update(status="error", error=str(e), quality=quality_episodes,
                                  flagged=flagged())
    finally:
        # The raw source is KEPT by default (keep_raw): re-uploading it from the
        # devices is the slow part of any rebuild, and a build cancelled from the
        # fleet must not leave a half-deleted source behind. Deleting it is now an
        # explicit opt-in from the caller, and even then only on success — a failed
        # run always keeps the raw so it can be inspected and retried.
        # (`api` may be unset if we failed before building it; the gates below
        # avoid touching it in that case.)
        if keep_raw:
            log(f"Kept raw dataset {source_repo} (raw datasets are not auto-deleted).")
        elif _rest_jobs[job_id].get("status") == "done":
            try:
                api.delete_repo(source_repo, repo_type="dataset")
                log(f"Deleted raw dataset {source_repo}.")
            except Exception as e:
                log(f"Warning: could not delete raw dataset: {e}")
        else:
            log(f"Kept raw dataset {source_repo} (run did not complete) — inspect or retry.")


@fastapi_app.post("/api/process")
async def api_process(req: _ProcessReq, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <hf_token>")
    token = authorization[len("Bearer "):].strip()
    job_id = uuid.uuid4().hex[:12]
    target_repo = req.target_repo.strip()
    # check_only is the explicit path, and the fleet now sends it: the device relay
    # forwards it from the fleet's process_dataset command. The target repo NAME is
    # kept as a fallback because it still arrives from any Pi running a build older
    # than that change — and unlike the flag, the name travels without a redeploy.
    # Drop the name-matching once no such device is in the field.
    name = target_repo.split("/")[-1]
    check_only = (req.check_only or SLAM_CHECK_MARKER in name
                  or name.startswith(SLAM_CHECK_PREFIX))
    _rest_jobs[job_id] = {"status": "running", "log": "", "error": None, "result": None,
                          "quality": None, "flagged": None, "visualizer": None,
                          # What /api/quality looks up: the caller behind the device
                          # knows the repo it asked for, never this job's id.
                          "target_repo": target_repo, "check_only": check_only}
    t = threading.Thread(
        target=_run_pipeline_headless,
        args=(job_id, req.source_repo.strip(), target_repo,
              req.task.strip(), token),
        kwargs={"exclude_fail": req.exclude_fail, "exclude_bad": req.exclude_bad,
                "private": req.private, "run_sync": req.run_sync,
                "tactile": req.tactile, "roles": req.roles,
                "keep_raw": req.keep_raw, "check_only": check_only},
        daemon=True,
    )
    t.start()
    return {"job_id": job_id}


@fastapi_app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    job = _rest_jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return job


@fastapi_app.get("/api/quality/{repo_id:path}")
async def api_quality(repo_id: str):
    """The latest run for a TARGET REPO, addressed by repo id instead of job id.

    Same in-memory record as /api/status (so the same unauthenticated exposure),
    but reachable by a caller that never learns the job id: the fleet has a DEVICE
    trigger the conversion, and the device reports back only the result. This is
    how a per-episode report reaches the fleet dashboard with no HF token in it and
    nothing to redeploy on the devices. `quality` is published live, so polling
    this streams verdicts as SLAM works through the set.
    """
    repo = repo_id.strip("/")
    match = None
    for job in _rest_jobs.values():  # insertion-ordered → the last match is the newest
        if job.get("target_repo") == repo:
            match = job
    if match is None:
        # Also the honest answer after a restart: the job list lives in memory only.
        return {"status": "not_found", "target_repo": repo}
    log = match.get("log") or ""
    return {"status": match["status"], "target_repo": repo,
            "check_only": match.get("check_only", False),
            "quality": match.get("quality") or [],
            "flagged": match.get("flagged"),
            "result": match.get("result"), "visualizer": match.get("visualizer"),
            "error": match.get("error"),
            # Last log line only: enough to show what the run is doing without
            # shipping the whole (long) log on every poll.
            "log_tail": log.rstrip().split("\n")[-1] if log.strip() else ""}


# Mount Gradio onto the FastAPI app. queue() is required for streaming generators
# and for the Stop button's cancels=[] to take effect.
def _without_session_cookie(headers):
    """The request headers with the `session` cookie removed, others untouched."""
    out = []
    for name, value in headers:
        if name.lower() == b"cookie":
            kept = [c for c in value.split(b";") if not c.strip().startswith(b"session=")]
            if not kept:
                continue
            value = b";".join(kept)
        out.append((name, value))
    return out


class _FreshLoginSession:
    """Start every login handshake from an empty session.

    Signing in twice used to loop forever between /login/huggingface and
    /login/callback. The session cookie still held `oauth_info` from the previous
    login — ~3.3 kB of access/refresh/id tokens — and starting a new login wrote
    that again next to the fresh `_state_huggingface_*`, landing at 4139 bytes.
    Browsers store at most 4096, so they dropped the Set-Cookie silently; the
    callback then arrived with the OLD cookie, which carried no state, and gradio
    bounced it back to the login route. Gradio's own recovery only fires when the
    cookie cannot be DECODED — this one decoded fine — so nothing broke the cycle
    and the only way out was deleting the cookie by hand.

    Dropping the cookie on the way in makes SessionMiddleware start a new session,
    so the handshake carries the state alone (~90 bytes). Re-authenticating
    supersedes the old token anyway. Only /login/huggingface is stripped: the
    callback must keep its cookie, that is where the state is read back from.

    Why the cookie and not `request.session`: gradio calls attach_oauth() inside
    App.create_app(), so SessionMiddleware lives on the INNER app that
    mount_gradio_app() mounts at "/". Anything wrapping the outer app runs before
    that middleware and has no scope["session"] — touching it raises
    "SessionMiddleware must be installed to access request.session".

    Plain ASGI rather than @app.middleware("http") for the same reason, and
    because BaseHTTPMiddleware interferes with the streaming responses gradio's
    queue depends on.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/login/huggingface":
            scope = dict(scope, headers=_without_session_cookie(scope["headers"]))
        await self.app(scope, receive, send)


# run_history=False: gradio 6 defaults it True and records every API call into the
# browser's localStorage. The review panel polls on a gr.Timer(0.5), so that is two
# entries a second for as long as a tab is open — unbounded growth against a ~5 MB
# per-origin quota, for calls nobody reviews. Switching it off also clears what the
# app has already saved in visitors' browsers.
app = gr.mount_gradio_app(fastapi_app, demo.queue(), path="/", theme=THEME, css=CSS,
                          run_history=False)
app = _FreshLoginSession(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
