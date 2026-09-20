"""tests/live/classify.py — OFFLINE tests of the live classifier check's logic (scripted fake classifier, no server)."""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest

from schemas.task_classification import TaskType
from tests.live import classify as C

ALL = C.CASES


def fake(answer_for):
    """A classify_fn: answer_for(task) -> task_type string, or an Exception to raise."""
    def classify(task, run_dir):
        a = answer_for(task)
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(task_type=a, confidence="high", reasoning=f"because {getattr(a, 'value', a)}")
    return classify


def go(classify, repeat=1, verbose=False, cases=ALL):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = C.run(cases, repeat, verbose, classify)
    return rc, buf.getvalue()


def perfect(task):
    return next(iter(next(c[2] for c in ALL if c[1] == task)))


# ── project-root discovery (the bug that made every case fail with ModuleNotFoundError) ──

def test_root_is_found_from_a_nested_directory(tmp_path):
    (tmp_path / "clients").mkdir()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "tests" / "deep").mkdir(parents=True)
    assert C.find_project_root(tmp_path / "tests" / "deep") == tmp_path
    assert C.find_project_root(tmp_path) == tmp_path


def test_root_search_fails_clearly_when_there_is_no_project(tmp_path):
    with pytest.raises(FileNotFoundError, match="project root"):
        C.find_project_root(tmp_path)


def test_a_directory_with_only_one_of_the_markers_is_not_the_root(tmp_path):
    (tmp_path / "clients").mkdir()
    with pytest.raises(FileNotFoundError):
        C.find_project_root(tmp_path)


def test_this_script_resolved_the_real_project_root(project_root):
    assert C.PROJECT_ROOT == project_root


# ── reporting ────────────────────────────────────────────────────────────────

def test_perfect_classifier_passes_everything():
    rc, out = go(fake(perfect))
    assert rc == 0 and f"{len(ALL)}/{len(ALL)} passed (100%)" in out


def test_the_original_search_bug_is_reported_as_a_failure():
    rc, out = go(fake(lambda t: "coding" if t == "search for today's date" else perfect(t)), verbose=True)
    assert rc == 1
    assert "[FAIL] 0/1  want describe" in out and "search for today's date" in out
    assert "because coding" in out                      # -v shows the model's reasoning


def test_reasoning_is_hidden_without_verbose():
    rc, out = go(fake(lambda t: "coding" if t == "search for today's date" else perfect(t)))
    assert "because coding" not in out and "re-run with -v" in out


def test_flaky_model_is_shown_as_partial_not_hidden_by_a_lucky_pass():
    n = {"i": 0}

    def flaky(task):
        n["i"] += 1
        return "describe" if n["i"] % 2 else "coding"
    rc, out = go(fake(flaky), repeat=2)
    assert "[part] 1/2" in out and rc == 1


def test_enum_like_task_type_values_are_unwrapped():
    enum_like = SimpleNamespace(value="describe")
    _, out = go(fake(lambda t: enum_like), cases=[c for c in ALL if c[2] == {"describe"}][:1])
    assert "got describe" in out


def test_accepted_sets_allow_the_fuzzy_cases():
    fuzzy = next(c for c in ALL if c[1].startswith("explore ways to add response caching"))
    assert {"mixed", "coding"} <= fuzzy[2]
    rc, _ = go(fake(lambda t: "coding"), cases=[fuzzy])
    assert rc == 0


# ── systemic failure vs per-case failure ─────────────────────────────────────

def test_identical_error_on_the_first_case_aborts_the_whole_run():
    with pytest.raises(C.SystemicFailure, match="ConnectionError"):
        go(fake(lambda t: ConnectionError("model server down")), repeat=3)


def test_error_on_one_later_case_is_reported_per_case_not_systemic():
    boom = ALL[3][1]
    rc, out = go(fake(lambda t: RuntimeError("one-off") if t == boom else perfect(t)))
    assert rc == 1 and out.count("ERROR: RuntimeError") == 1
    assert f"{len(ALL) - 1}/{len(ALL)} passed" in out


def test_first_case_erroring_differently_each_time_is_not_treated_as_systemic():
    n = {"i": 0}

    def varied(task):
        n["i"] += 1
        return RuntimeError(f"different {n['i']}") if task == ALL[0][1] else perfect(task)
    rc, out = go(fake(varied), repeat=2)
    assert rc == 1 and "ERROR: RuntimeError" in out


# ── the case list itself ─────────────────────────────────────────────────────

def test_every_accepted_type_is_a_real_task_type():
    valid = {t.value for t in TaskType}
    for group, task, accepted, note in ALL:
        assert accepted <= valid, f"{task!r} accepts unknown type(s): {accepted - valid}"


def test_tasks_are_unique_and_cover_both_sides_of_the_search_boundary():
    tasks = [c[1] for c in ALL]
    assert len(tasks) == len(set(tasks))
    groups = {c[0] for c in ALL}
    assert {"search -> describe", "search -> coding", "search -> ideation/mixed", "unchanged"} <= groups


# ── truncation detail, cap, and classify_node parity ─────────────────────────

from clients.llm import TruncatedOutputError  # noqa: E402


def truncation(thinking="", partial="", cap=2000):
    return TruncatedOutputError(stage="classify", cap=cap, tokens_out=cap, thinking_block=thinking, partial_answer=partial)


def test_truncation_error_reports_what_the_model_spent_its_tokens_on():
    label, detail = C.describe_error(truncation(thinking="Wait, let me reconsider. " * 40))
    assert label == "ERROR: TruncatedOutputError"
    assert "response truncated at max_tokens=2000" in detail
    assert "chars of THINKING" in detail and "0 chars of ANSWER" in detail
    assert "thinking: Wait, let me reconsider." in detail


def test_long_thinking_is_shown_as_head_and_tail_so_a_loop_is_visible():
    text = "START " + "wait " * 500 + "THE END"
    detail = C.describe_error(truncation(thinking=text))[1]
    assert "START" in detail and "THE END" in detail and "[...]" in detail
    assert len(detail) < 1200


def test_answer_excerpt_is_shown_when_the_json_itself_rambles():
    detail = C.describe_error(truncation(partial='{"mode": "short", "reasoning": "' + "blah " * 300))[1]
    assert "0 chars of THINKING" in detail and "answer:" in detail


def test_ordinary_errors_get_no_excerpt_lines():
    label, detail = C.describe_error(ConnectionError("refused"))
    assert label == "ERROR: ConnectionError" and detail == "refused"


def test_repeated_truncation_is_systemic_even_though_the_excerpts_differ_each_time():
    n = {"i": 0}

    def classify(task, run_dir):
        n["i"] += 1
        raise truncation(thinking=f"different rambling number {n['i']}")
    with pytest.raises(C.SystemicFailure) as exc:
        go(classify, repeat=3)
    assert "THINKING" in str(exc.value)


def test_truncation_on_one_later_case_is_reported_per_case():
    target = ALL[4][1]
    rc, out = go(fake(lambda t: truncation(thinking="x") if t == target else perfect(t)), verbose=True)
    assert rc == 1 and out.count("ERROR: TruncatedOutputError") >= 1 and "chars of THINKING" in out


def test_real_classifier_mirrors_classify_node_and_applies_the_cap(monkeypatch):
    import clients.llm as llm_mod
    seen = {}
    monkeypatch.setattr(llm_mod, "call_role", lambda **kw: seen.update(kw) or SimpleNamespace(task_type="describe", confidence="high"))
    C.real_classifier("hello", "/tmp", cap=800)
    assert seen["output_cap_override"] == 800
    assert seen["max_retries"] == 0 and seen["thinking"] is False and seen["allow_escalation"] is False
    assert seen["role"] == "classify" and seen["template_vars"] == {"task": "hello"}


def test_default_cap_is_far_below_the_routing_cap_so_runaways_fail_fast():
    assert C.DEFAULT_CAP <= 2000


def test_main_explains_a_truncation_and_points_at_the_diagnostic(monkeypatch, capsys):
    monkeypatch.setattr(C, "run", lambda *a, **k: (_ for _ in ()).throw(C.SystemicFailure("ERROR: TruncatedOutputError: truncated")))
    monkeypatch.setattr("sys.argv", ["classify.py"])
    assert C.main() == 2
    err = capsys.readouterr().err
    assert "tests/live/thinking_control.py" in err and "THINKING figure" in err


def test_main_passes_the_cap_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(C, "real_classifier", lambda task, run_dir, cap: seen.setdefault("cap", cap) and SimpleNamespace(task_type="describe", confidence="high", reasoning=""))
    monkeypatch.setattr("sys.argv", ["classify.py", "--cap", "777"])
    C.main()
    assert seen["cap"] == 777