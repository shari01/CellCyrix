Run (from repo root, venv on, key in .env or env):
```
RUN_LIVE_LLM_TESTS=1 pytest cellcyrix/llm/tests/test_openrouter_live.py -m integration -v
```
Cheaper run (skip heavy and agent primaries):
```
RUN_LIVE_LLM_TESTS=1 RUN_LIVE_LLM_EXCLUDE=heavy,agent pytest cellcyrix/llm/tests/test_openrouter_live.py -m integration -v
```
pytest marker integration is registered in pyproject.toml. The normal suite still gives 23 passed, 6 skipped when RUN_LIVE_LLM_TESTS is unset.