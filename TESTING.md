# Testing

Two kinds of checks, kept in two separate folders so they can't be confused:

|                        | `tests/`                                   | `tests/live/`                                  |
|------------------------|--------------------------------------------|-------------------------------------------------|
| Needs a running model? | **No** — fully offline                     | **Yes** — talks to your llama-server            |
| Run with               | `pytest`                                   | `python tests/live/<script>.py`                |
| Takes                  | ~3 seconds                                 | seconds to a couple of minutes                  |
| Answers                | "Does the code still behave?" (regressions)| "Does the real model / server behave?"          |

`pytest` only ever collects `tests/test_*.py`; it never touches `tests/live/`.

## Offline tests (run these constantly)

```bash
pip install -r requirements-test.txt        # once: pytest + pytest-cov

pytest                                      # everything
pytest tests/test_describe_node.py          # one file
pytest -k "recovery"                        # tests whose name matches
pytest -x                                   # stop at the first failure
pytest --lf                                 # re-run only what failed last time

pytest --cov --cov-report=term-missing      # coverage, with uncovered line numbers
pytest --cov --cov-report=html              # then open htmlcov/index.html
```

Run from the project root (`pytest.ini` and `.coveragerc` live there). The overall coverage figure
includes large parts of the codebase these tests don't target, so read the per-file lines.

## Live checks (run these after touching a prompt or a llama-server flag)

Both need a model. `classify.py` goes through the pipeline's own model loading, like a real run;
`thinking_control.py` talks to the server directly, so start the 9B first
(`curl http://localhost:8081/health` should answer).

```bash
python tests/live/thinking_control.py      # ~1 min: which "thinking off" switch does your server honour?
python tests/live/classify.py              # is classify.yaml routing tasks correctly? (one pass)
python tests/live/classify.py -n 3 -v      # 3 passes per case (the 9B is non-deterministic) + reasons for misses
python tests/live/classify.py --cap 800    # abort any classification that generates more than 800 tokens
```

Reach for these when: you edit `config/prompts/classify.yaml`, change llama-server flags, or a
"non-thinking" stage (classify, validate, audit, bugfix) starts running suspiciously long.

## Layout

```
tests/         offline pytest suite (test_*.py, conftest.py, fakes.py)
tests/live/   scripts that need a live server — never collected by pytest
pytest.ini     collects tests/test_*.py only
.coveragerc    what `--cov` measures
```

The scripts' own logic (argument handling, pass/fail reporting, the analysis text) is unit-tested
offline in `tests/test_tests/live_classify.py` and `tests/test_tests/live_thinking.py`.