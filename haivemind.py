#!/usr/bin/env python3
"""hAIvemind: multi-model consensus loop via OpenCode CLI.

MVP goals:
- No "arbiter" model; consensus emerges when drafts converge.
- Shared memory on disk (run directory with logs + drafts).
- Works with any models reachable via `opencode run -m <provider/model>`.

This is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOM_RULES = """\
You are one of several AI agents locked in a room with other agents.

Goal:
- Solve the user's problem and produce the best possible final answer.
- Nobody leaves until you all agree on a CORRECT, high-quality answer.

How to behave:
- Read what the other agents wrote.
- Independently verify claims; do not agree just to agree.
- Attack weak reasoning with concrete counterexamples or evidence.
- If you are outnumbered and cannot produce new evidence, concede and adopt the better approach.
- If the group is converging too fast, be the skeptic/breaker and try to disprove the emerging consensus.
- If discussion is stuck, be the synthesizer: propose a merged resolution that addresses objections.

You are not assigned a fixed role. Fluidly adopt what the conversation needs:
- CRITIC, BUILDER, EXPERT, SKEPTIC, SYNTHESIZER, BREAKER, CONCEDER
"""


JSON_INSTRUCTIONS = """\
Return ONLY a valid JSON object (no markdown fences, no commentary).

Schema:
{
  "draft": string,               // your best current final answer, standalone
  "key_points": string[],        // 3-10 bullets of the core content
  "disagreements": string[],     // list of remaining disagreements you see (empty if none)
  "changes_made": string[],      // what you changed vs your previous position (empty in round 1)
  "confidence": number           // 0.0 - 1.0
}
"""

# ---------------------------------------------------------------------------
# Default number of consecutive failures before a model is dropped mid-run.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 2

# Default number of retries when a model returns invalid JSON within a round.
_DEFAULT_JSON_RETRIES = 1


def _now_slug() -> str:
    # 2026-02-26_23-30-59
    return _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _slugify(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        return "prompt"
    return s[:max_len].strip("-")


def _ratio(a: str, b: str) -> float:
    # SequenceMatcher is cheap and good enough for a convergence heuristic.
    return difflib.SequenceMatcher(a=a, b=b).ratio()


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)\n```", re.DOTALL)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # If the whole output is a single fenced block, unwrap it.
    m = _FENCE_RE.fullmatch(s)
    if m:
        return (m.group(1) or "").strip()
    return s


def _normalize_for_similarity(s: str) -> str:
    # Normalize superficial formatting so equivalent drafts converge faster.
    s = _strip_code_fences(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    if not s:
        return ""
    # Collapse runs of whitespace to a single space.
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _extract_json_maybe(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    # Unwrap a single fenced block (common for markdown-ish models).
    text2 = _strip_code_fences(text)
    if text2:
        text = text2

    # Fast path: whole payload is JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: parse the first JSON object embedded in text.
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = dec.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    return None


def _run_opencode(
    *,
    opencode_bin: str,
    opencode_agent: str | None,
    model: str,
    prompt: str,
    timeout_s: int,
    cwd: Path,
    live: bool,
    on_text,
) -> tuple[str, str]:
    """Run `opencode run` and return (stdout_text, stderr_text)."""
    cmd = [opencode_bin, "run", "-m", model]
    if opencode_agent:
        cmd += ["--agent", opencode_agent]
    cmd += ["--format", "json", prompt]

    # Live mode: stream JSONL and print "text" fragments as they arrive.
    if live:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_stream = proc.stdout
        stderr_stream = proc.stderr
        if stdout_stream is None or stderr_stream is None:
            proc.kill()
            raise RuntimeError("Failed to open opencode stdout/stderr pipes")

        parts: list[str] = []
        stderr_parts: list[str] = []
        opencode_errors: list[str] = []

        def _stderr_reader() -> None:
            try:
                for l in stderr_stream:
                    stderr_parts.append(l)
            except Exception:
                return

        t = threading.Thread(target=_stderr_reader, daemon=True)
        t.start()

        done = threading.Event()
        timed_out = threading.Event()

        def _watchdog() -> None:
            if not timeout_s:
                return
            if done.wait(timeout=float(timeout_s)):
                return
            timed_out.set()
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                return

        w = threading.Thread(target=_watchdog, daemon=True)
        w.start()

        for raw_line in stdout_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue

            if evt.get("type") == "text":
                part = evt.get("part") or {}
                frag = part.get("text")
                if isinstance(frag, str) and frag:
                    parts.append(frag)
                    try:
                        on_text(frag)
                    except Exception:
                        pass
            elif evt.get("type") == "error":
                err = evt.get("error") or {}
                data = err.get("data") or {}
                msg = data.get("message")
                if not isinstance(msg, str) or not msg.strip():
                    msg = json.dumps(data) if data else json.dumps(err) if err else line
                msg = msg.strip()
                opencode_errors.append(msg)
                try:
                    on_text(f"\n[opencode error] {msg}\n")
                except Exception:
                    pass

        try:
            proc.wait(timeout=max(1, int(timeout_s) if timeout_s else 10))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        done.set()
        try:
            w.join(timeout=0.2)
        except Exception:
            pass

        try:
            t.join(timeout=0.2)
        except Exception:
            pass

        stderr_text = "".join(stderr_parts).strip()
        if opencode_errors:
            extra = "\n".join(f"opencode_error: {m}" for m in opencode_errors)
            stderr_text = (stderr_text + "\n" + extra).strip() if stderr_text else extra

        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)

        return ("".join(parts).strip(), stderr_text)

    # Non-live mode: capture everything and parse at the end.
    proc2 = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    # opencode emits JSONL; stdout may also include non-JSON lines if something goes sideways.
    parts2: list[str] = []
    opencode_errors2: list[str] = []
    for line in proc2.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except Exception:
            continue

        if evt.get("type") == "text":
            part = evt.get("part") or {}
            t2 = part.get("text")
            if isinstance(t2, str) and t2:
                parts2.append(t2)
        elif evt.get("type") == "error":
            err = evt.get("error") or {}
            data = err.get("data") or {}
            msg = data.get("message")
            if not isinstance(msg, str) or not msg.strip():
                msg = json.dumps(data) if data else json.dumps(err) if err else line
            opencode_errors2.append(msg.strip())

    stderr2 = proc2.stderr.strip()
    if opencode_errors2:
        extra2 = "\n".join(f"opencode_error: {m}" for m in opencode_errors2)
        stderr2 = (stderr2 + "\n" + extra2).strip() if stderr2 else extra2

    return ("".join(parts2).strip(), stderr2)


@dataclass
class AgentRound:
    model: str
    raw_text: str
    parsed: dict[str, Any] | None
    error: str | None

    @property
    def draft(self) -> str:
        if self.parsed and isinstance(self.parsed.get("draft"), str):
            return self.parsed["draft"].strip()
        # Last resort: treat raw as a draft.
        return (self.raw_text or "").strip()

    @property
    def has_valid_draft(self) -> bool:
        """True when the model returned parseable JSON with a draft field."""
        return (
            self.parsed is not None
            and isinstance(self.parsed.get("draft"), str)
            and bool(self.parsed["draft"].strip())
        )

    @property
    def confidence(self) -> float:
        """Return the model's self-reported confidence, defaulting to 0.5."""
        if self.parsed and isinstance(self.parsed.get("confidence"), (int, float)):
            c = float(self.parsed["confidence"])
            return max(0.0, min(1.0, c))
        return 0.5


def _build_agent_prompt(
    *,
    user_prompt: str,
    models: list[str],
    round_index: int,
    history: list[dict[str, Any]],
) -> str:
    # Keep the prompt compact but informative. Provide prior drafts so agents can converge.
    header = f"Round {round_index} / Shared Room\n"

    history_block = ""
    if history:
        # Only include the last round's drafts (keeps token usage bounded).
        last = history[-1]
        drafts = last.get("drafts") or {}
        lines: list[str] = []
        for m in models:
            d = drafts.get(m, "")
            if not isinstance(d, str):
                d = ""
            d = d.strip()
            if not d:
                continue
            clipped = d
            if len(clipped) > 6000:
                clipped = clipped[:6000] + "\n\n[...clipped...]"
            lines.append(f"=== {m} (last draft) ===\n{clipped}\n")
        if lines:
            history_block = "\n".join(lines)

    prompt = "\n".join(
        [
            header,
            "ROOM RULES:\n" + ROOM_RULES.strip(),
            "\nTASK (user prompt):\n" + user_prompt.strip(),
            ("\nPREVIOUS ROUND DRAFTS:\n" + history_block) if history_block else "",
            "\n" + JSON_INSTRUCTIONS.strip(),
        ]
    ).strip()

    return prompt


def _find_consensus(
    drafts_by_model: dict[str, str],
    *,
    threshold: float,
    min_agree: int,
    basis_by_model: dict[str, str] | None = None,
    key_points_by_model: dict[str, list[str]] | None = None,
    confidence_by_model: dict[str, float] | None = None,
) -> tuple[list[str], str] | None:
    models = [m for m, d in drafts_by_model.items() if d.strip()]
    if len(models) < min_agree:
        return None

    if basis_by_model is None:
        basis_by_model = drafts_by_model
    norm = {
        m: _normalize_for_similarity(str(basis_by_model.get(m, "") or ""))
        for m in models
    }

    def _norm_kps(m: str) -> str:
        if not key_points_by_model:
            return ""
        kps = key_points_by_model.get(m)
        if not isinstance(kps, list) or not kps:
            return ""
        cleaned: list[str] = []
        for kp in kps:
            if not isinstance(kp, str):
                continue
            t = _normalize_for_similarity(kp)
            if t:
                cleaned.append(t)
        if not cleaned:
            return ""
        cleaned = sorted(set(cleaned))
        return "\n".join(cleaned)

    norm_kp = {m: _norm_kps(m) for m in models}

    # Build adjacency: i ~ j if similarity >= threshold
    sim: dict[tuple[str, str], float] = {}
    for i, mi in enumerate(models):
        for mj in models[i + 1 :]:
            r = _ratio(norm[mi], norm[mj])
            if norm_kp[mi] and norm_kp[mj]:
                r = max(r, _ratio(norm_kp[mi], norm_kp[mj]))
            sim[(mi, mj)] = r
            sim[(mj, mi)] = r

    groups: list[list[str]] = []
    for mi in models:
        group = [mi]
        for mj in models:
            if mj == mi:
                continue
            if sim.get((mi, mj), 0.0) >= threshold:
                group.append(mj)
        if len(group) >= min_agree:
            group = sorted(set(group))
            groups.append(group)

    if not groups:
        return None

    # Pick the largest group; break ties by medoid score.
    groups.sort(key=lambda g: (-len(g), ",".join(g)))
    best_group = groups[0]

    # Confidence weighting: when selecting the medoid, factor in each model's
    # self-reported confidence so higher-confidence models are preferred when
    # similarity is otherwise close.
    conf = confidence_by_model or {}

    def avg_sim(m: str) -> float:
        others = [o for o in best_group if o != m]
        if not others:
            return 1.0
        raw = sum(sim.get((m, o), _ratio(norm[m], norm[o])) for o in others) / len(
            others
        )
        # Blend in confidence as a small bonus (up to 5%) so it acts as a
        # tiebreaker rather than overriding similarity.
        c = conf.get(m, 0.5)
        return raw + 0.05 * c

    medoid = max(best_group, key=avg_sim)
    return (best_group, drafts_by_model[medoid].strip())


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _probe_models(
    *,
    opencode_bin: str,
    opencode_agent: str | None,
    models: list[str],
    timeout_s: int,
    retries: int,
    parallel: int,
    cwd: Path,
    live: bool,
    live_write,
) -> tuple[list[str], dict[str, str]]:
    """Return (healthy_models, failures_by_model)."""

    # Probe is about reachability, not strict instruction-following.
    # Some providers/models can be healthy but respond with extra formatting.
    probe_prompt = "Reply with exactly: OK"

    failures: dict[str, str] = {}
    healthy: list[str] = []

    def check(m: str) -> tuple[str, bool, str]:
        last_reason = ""
        raw = ""
        stderr = ""
        for attempt in range(1, max(0, int(retries)) + 2):
            last_reason = ""
            try:
                raw, stderr = _run_opencode(
                    opencode_bin=opencode_bin,
                    opencode_agent=opencode_agent,
                    model=m,
                    prompt=probe_prompt,
                    timeout_s=timeout_s,
                    cwd=cwd,
                    live=False,
                    on_text=(lambda _frag: None),
                )
            except subprocess.TimeoutExpired:
                last_reason = f"timeout after {timeout_s}s"
            except FileNotFoundError:
                last_reason = f"opencode not found: {opencode_bin}"
                break
            except Exception as e:
                last_reason = f"error: {e}"

            if not last_reason:
                if stderr:
                    last_reason = stderr
                elif not raw.strip():
                    last_reason = "empty probe output"
                else:
                    return (m, True, "")

            if attempt <= int(retries):
                time.sleep(0.4 * attempt)

        return (m, False, last_reason or "probe_failed")

    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as ex:
        futs = {ex.submit(check, m): m for m in models}
        for fut in as_completed(futs):
            m, ok, reason = fut.result()
            if ok:
                healthy.append(m)
                if live:
                    live_write(f"[probe] ok: {m}\n")
            else:
                failures[m] = reason
                if live:
                    live_write(f"[probe] fail: {m}: {reason.strip().splitlines()[0]}\n")

    healthy.sort(key=lambda x: models.index(x))
    return (healthy, failures)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="haivemind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            hAIvemind: bounce a prompt across multiple models until their drafts converge.

            Example:
              python haivemind.py \\
                --models google/gemini-3-flash-preview,google/gemini-2.5-flash,qwen/qwen3-coder-plus \\
                "Write a robust bash function to ..."
            """
        ),
    )
    parser.add_argument("prompt", nargs="+", help="User prompt")
    parser.add_argument(
        "--models",
        default=(
            "google/gemini-3-flash-preview,google/gemini-2.5-flash,"
            "qwen/qwen3-coder-plus,nvidia/meta/llama3-8b-instruct"
        ),
        help="Comma-separated provider/model list",
    )
    parser.add_argument(
        "--allow-single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue even if only 1 healthy model remains after probe (default: true)"
        ),
    )
    parser.add_argument("--rounds", type=int, default=5, help="Max rounds")
    parser.add_argument(
        "--min-agree", type=int, default=3, help="Min models that must match to stop"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.92, help="Draft similarity threshold (0-1)"
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="Per-model timeout seconds"
    )
    parser.add_argument(
        "--parallel", type=int, default=3, help="Max concurrent model calls"
    )
    parser.add_argument(
        "--outdir",
        default="tmp/haivemind",
        help="Base output directory (run artifacts stored under a timestamped subdir)",
    )
    parser.add_argument(
        "--opencode-bin",
        default=os.environ.get("OPENCODE_BIN", "opencode"),
        help="Path to opencode binary (default: OPENCODE_BIN or 'opencode')",
    )
    parser.add_argument(
        "--opencode-agent",
        default=os.environ.get("OPENCODE_AGENT", ""),
        help="OpenCode agent name (default: OPENCODE_AGENT env var, or none)",
    )
    parser.add_argument(
        "--probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preflight models and drop ones that error (default: true)",
    )
    parser.add_argument(
        "--probe-timeout",
        type=int,
        default=60,
        help="Per-model probe timeout seconds",
    )
    parser.add_argument(
        "--probe-retries",
        type=int,
        default=1,
        help="Probe retries per model (default: 1)",
    )
    parser.add_argument(
        "--json-retries",
        type=int,
        default=_DEFAULT_JSON_RETRIES,
        help=(
            "Retries per model when it returns invalid JSON within a round "
            f"(default: {_DEFAULT_JSON_RETRIES})"
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=_DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=(
            "Drop a model after this many consecutive round failures "
            f"(default: {_DEFAULT_MAX_CONSECUTIVE_FAILURES})"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream per-model outputs live to stderr while running",
    )
    parser.add_argument(
        "--live-compact",
        action="store_true",
        help="When --live, prefix fragments with model id",
    )
    parser.add_argument(
        "--show-log", action="store_true", help="Print run artifact paths to stderr"
    )
    args = parser.parse_args(argv)

    user_prompt = " ".join(args.prompt).strip()
    models_requested = [m.strip() for m in str(args.models).split(",") if m.strip()]
    models = list(models_requested)
    if len(set(models)) < 2 and not bool(args.allow_single):
        print(
            "Need at least 2 models for consensus (or pass --allow-single).",
            file=sys.stderr,
        )
        return 2

    base_out = Path(args.outdir)
    run_dir = base_out / f"{_now_slug()}_{_slugify(user_prompt)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "prompt.txt").write_text(user_prompt + "\n", encoding="utf-8")
    (run_dir / "room_rules.txt").write_text(ROOM_RULES.strip() + "\n", encoding="utf-8")
    (run_dir / "models.requested.txt").write_text(
        "\n".join(models_requested) + "\n", encoding="utf-8"
    )

    meta = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "models_requested": models_requested,
        "rounds_max": args.rounds,
        "min_agree": args.min_agree,
        "threshold": args.threshold,
        "timeout_s": args.timeout,
        "parallel": args.parallel,
        "opencode_bin": args.opencode_bin,
        "opencode_agent": args.opencode_agent,
        "probe": args.probe,
        "probe_timeout_s": args.probe_timeout,
        "probe_retries": args.probe_retries,
        "json_retries": args.json_retries,
        "max_consecutive_failures": args.max_consecutive_failures,
    }
    _write_json(run_dir / "meta.json", meta)

    history: list[dict[str, Any]] = []
    consensus: tuple[list[str], str] | None = None

    live_lock = threading.Lock()

    def live_write(model: str, s: str) -> None:
        if not args.live:
            return
        if not s:
            return
        with live_lock:
            if args.live_compact:
                sys.stderr.write(f"[{model}] {s}")
            else:
                sys.stderr.write(s)
            sys.stderr.flush()

    # Preflight models so "only one model" doesn't silently happen.
    opencode_agent = str(args.opencode_agent).strip() or None
    if args.probe:
        if args.live:
            with live_lock:
                sys.stderr.write("[probe] checking models...\n")
                sys.stderr.flush()

        healthy, failures = _probe_models(
            opencode_bin=args.opencode_bin,
            opencode_agent=opencode_agent,
            models=models,
            timeout_s=int(args.probe_timeout),
            retries=int(args.probe_retries),
            parallel=int(args.parallel),
            cwd=run_dir,
            live=bool(args.live),
            live_write=(lambda msg: live_write("probe", msg)),
        )
        _write_json(run_dir / "probe.json", {"healthy": healthy, "failures": failures})
        models = healthy

        (run_dir / "models.active.txt").write_text(
            "\n".join(models) + "\n", encoding="utf-8"
        )

        if len(models) < 2 and not bool(args.allow_single):
            print(
                "Not enough healthy models after probe. See artifacts in: "
                + str(run_dir),
                file=sys.stderr,
            )
            return 1
        if len(models) < 2 and bool(args.allow_single):
            print(
                "Warning: only 1 healthy model after probe; continuing in single-model mode.",
                file=sys.stderr,
            )
    else:
        (run_dir / "models.active.txt").write_text(
            "\n".join(models) + "\n", encoding="utf-8"
        )

    if not models:
        print(
            "No active models to run. See artifacts in: " + str(run_dir),
            file=sys.stderr,
        )
        return 1

    effective_min_agree = min(int(args.min_agree), len(models))
    if effective_min_agree != int(args.min_agree):
        _write_json(
            run_dir / "min_agree.adjustment.json",
            {
                "requested": int(args.min_agree),
                "effective": effective_min_agree,
                "active_models": len(models),
            },
        )
        if args.live:
            with live_lock:
                sys.stderr.write(
                    f"[probe] adjusted --min-agree {int(args.min_agree)} -> {effective_min_agree}\n"
                )
                sys.stderr.flush()

    # -----------------------------------------------------------------------
    # Per-model consecutive failure tracker.  When a model fails this many
    # rounds in a row (invalid JSON, timeout, etc.) it gets dropped.
    # -----------------------------------------------------------------------
    consecutive_failures: dict[str, int] = {m: 0 for m in models}
    dropped_models: dict[str, str] = {}  # model -> reason

    json_retries = max(0, int(args.json_retries))
    max_consec = max(1, int(args.max_consecutive_failures))

    # Last round's results, kept outside the loop so the fallback can access it.
    results: dict[str, AgentRound] = {}

    for round_index in range(1, int(args.rounds) + 1):
        round_started = time.time()
        prompt_for_agents = _build_agent_prompt(
            user_prompt=user_prompt,
            models=models,
            round_index=round_index,
            history=history,
        )

        rounds_dir = run_dir / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        round_dir = rounds_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "agent_prompt.txt").write_text(
            prompt_for_agents + "\n", encoding="utf-8"
        )

        results.clear()

        def key_points_list(ar: AgentRound) -> list[str]:
            if not ar.parsed:
                return []
            kps = ar.parsed.get("key_points")
            if not isinstance(kps, list):
                return []
            return [kp for kp in kps if isinstance(kp, str)]

        def _call_opencode_once(
            m: str,
            prompt: str,
            round_idx: int,
            rdir: Path,
        ) -> AgentRound:
            """Single attempt to call a model and parse the result."""
            try:
                if args.live and not args.live_compact:
                    with live_lock:
                        sys.stderr.write(f"\n=== {m} (round {round_idx}) ===\n")
                        sys.stderr.flush()
                raw, stderr = _run_opencode(
                    opencode_bin=args.opencode_bin,
                    opencode_agent=opencode_agent,
                    model=m,
                    prompt=prompt,
                    timeout_s=int(args.timeout),
                    cwd=run_dir,
                    live=bool(args.live),
                    on_text=(lambda frag, mm=m: live_write(mm, frag)),
                )
                if args.live and not args.live_compact:
                    with live_lock:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
            except subprocess.TimeoutExpired:
                return AgentRound(
                    model=m,
                    raw_text="",
                    parsed=None,
                    error=f"timeout after {args.timeout}s",
                )
            except FileNotFoundError:
                return AgentRound(
                    model=m,
                    raw_text="",
                    parsed=None,
                    error=f"opencode not found: {args.opencode_bin}",
                )
            except Exception as e:
                return AgentRound(
                    model=m, raw_text="", parsed=None, error=f"error: {e}"
                )

            parsed = _extract_json_maybe(raw)
            err = None
            if parsed is None:
                err = "invalid_json"
            elif not isinstance(parsed.get("draft"), str):
                err = "missing_draft"

            if stderr and err is None:
                # Surface provider/API issues even if JSON parsed.
                err = "stderr"

            # Persist raw for theatre/debug.
            (rdir / f"{_slugify(m, max_len=80)}.raw.txt").write_text(
                raw + "\n", encoding="utf-8"
            )
            if stderr:
                (rdir / f"{_slugify(m, max_len=80)}.stderr.txt").write_text(
                    stderr + "\n", encoding="utf-8"
                )
            if parsed is not None:
                _write_json(rdir / f"{_slugify(m, max_len=80)}.json", parsed)

            return AgentRound(model=m, raw_text=raw, parsed=parsed, error=err)

        def call_model(m: str) -> AgentRound:
            """Call a model with optional JSON retries."""
            ar = _call_opencode_once(m, prompt_for_agents, round_index, round_dir)

            # Retry on invalid JSON / missing draft (not on timeouts or hard errors).
            retries_left = json_retries
            while retries_left > 0 and ar.error in ("invalid_json", "missing_draft"):
                retries_left -= 1
                if args.live:
                    with live_lock:
                        sys.stderr.write(
                            f"[{m}] retrying ({ar.error}, "
                            f"{json_retries - retries_left}/{json_retries})...\n"
                        )
                        sys.stderr.flush()
                ar = _call_opencode_once(m, prompt_for_agents, round_index, round_dir)

            return ar

        with ThreadPoolExecutor(max_workers=max(1, int(args.parallel))) as ex:
            futs = {ex.submit(call_model, m): m for m in models}
            for fut in as_completed(futs):
                m = futs[fut]
                results[m] = fut.result()

        # -------------------------------------------------------------------
        # Update consecutive failure counts and drop models that keep failing.
        # -------------------------------------------------------------------
        for m in list(models):
            ar = results.get(m)
            if ar and ar.has_valid_draft:
                consecutive_failures[m] = 0
            else:
                consecutive_failures[m] = consecutive_failures.get(m, 0) + 1
                if consecutive_failures[m] >= max_consec:
                    reason = (
                        f"dropped after {consecutive_failures[m]} consecutive failures"
                    )
                    dropped_models[m] = reason
                    if args.live:
                        with live_lock:
                            sys.stderr.write(f"[{m}] {reason}\n")
                            sys.stderr.flush()

        # Remove dropped models from the active list.
        if dropped_models:
            models = [m for m in models if m not in dropped_models]
            # Recalculate effective_min_agree after dropping.
            effective_min_agree = min(int(args.min_agree), len(models))

            # Persist updated active model list.
            (run_dir / "models.active.txt").write_text(
                "\n".join(models) + "\n", encoding="utf-8"
            )

        if not models:
            print(
                "All models dropped due to consecutive failures. "
                "See artifacts in: " + str(run_dir),
                file=sys.stderr,
            )
            # Still write history before bailing.
            _write_json(run_dir / "history.json", history)
            return 1

        # -------------------------------------------------------------------
        # Build drafts for consensus, preferring the parsed draft field over
        # raw text so that meta-summaries from misbehaving agents don't
        # pollute the comparison.
        # -------------------------------------------------------------------
        drafts_by_model: dict[str, str] = {}
        for m in models:
            if m not in results:
                continue
            ar = results[m]
            if ar.has_valid_draft:
                drafts_by_model[m] = ar.parsed["draft"].strip()  # type: ignore[index]
            else:
                # Fallback to raw, but only if there's something there.
                raw_d = (ar.raw_text or "").strip()
                if raw_d:
                    drafts_by_model[m] = raw_d

        consensus = _find_consensus(
            drafts_by_model,
            threshold=float(args.threshold),
            min_agree=effective_min_agree,
            basis_by_model=drafts_by_model,
            key_points_by_model={
                m: (key_points_list(results[m]) if m in results else []) for m in models
            },
            confidence_by_model={
                m: (results[m].confidence if m in results else 0.5) for m in models
            },
        )

        round_record: dict[str, Any] = {
            "round": round_index,
            "started_at": _dt.datetime.fromtimestamp(round_started).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(time.time() - round_started, 3),
            "drafts": drafts_by_model,
            "errors": {
                m: results[m].error
                for m in list(drafts_by_model) + list(dropped_models)
                if results.get(m) and results[m].error
            },
            "consensus": {
                "models": consensus[0],
                "threshold": args.threshold,
            }
            if consensus
            else None,
        }
        if dropped_models:
            round_record["dropped_models"] = dict(dropped_models)
        history.append(round_record)
        _write_json(run_dir / "history.json", history)

        if consensus:
            (run_dir / "final.md").write_text(consensus[1] + "\n", encoding="utf-8")
            (run_dir / "final.txt").write_text(consensus[1] + "\n", encoding="utf-8")
            break

        if args.live:
            with live_lock:
                sys.stderr.write(
                    f"\n--- end round {round_index} ({round_record['duration_s']}s) ---\n"
                )
                sys.stderr.flush()

    if not consensus:
        # Fallback: pick the most central draft from the last round.
        # Prefer models that produced valid JSON drafts.
        last = history[-1]["drafts"] if history else {}
        models_with_drafts = [
            m for m in models if isinstance(last.get(m), str) and last[m].strip()
        ]
        if not models_with_drafts:
            print(
                "No model produced a usable draft. See artifacts in: " + str(run_dir),
                file=sys.stderr,
            )
            return 1

        # Partition into valid-JSON models and raw-fallback models.
        valid_json_models = [
            m for m in models_with_drafts if m in results and results[m].has_valid_draft
        ]
        # Prefer valid JSON models for centrality; fall back to all if none.
        candidate_pool = valid_json_models if valid_json_models else models_with_drafts

        def centrality(m: str) -> float:
            d = last[m]
            others = [o for o in candidate_pool if o != m]
            if not others:
                return 1.0
            base = sum(_ratio(d, last[o]) for o in others) / len(others)
            # Small confidence bonus as tiebreaker.
            c = results[m].confidence if m in results else 0.5
            return base + 0.05 * c

        best = max(candidate_pool, key=centrality)
        draft = str(last[best]).strip()
        (run_dir / "final.md").write_text(draft + "\n", encoding="utf-8")
        (run_dir / "final.txt").write_text(draft + "\n", encoding="utf-8")

        if args.show_log:
            print(
                f"No hard consensus; chose central draft from: {best}", file=sys.stderr
            )

        print(draft)
        if args.show_log:
            print(f"Artifacts: {run_dir}", file=sys.stderr)
        return 0

    # Success: print final answer.
    final_text = consensus[1].strip()
    print(final_text)
    if args.show_log:
        print(f"Consensus models: {', '.join(consensus[0])}", file=sys.stderr)
        print(f"Artifacts: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
