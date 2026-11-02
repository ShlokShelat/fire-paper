# Offline entry points. Nothing here calls a model or needs an API key.
# Use another interpreter with: make test PYTHON=/path/to/python

PYTHON ?= python3
# Run every offline target with the API keys removed from the environment.
OFFLINE = env -u OPENAI_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY -u XAI_API_KEY \
	-u GROK_API_KEY HF_HUB_OFFLINE=1

SELFTESTS = \
	src/preprocessing/fire_preprocessor.py \
	src/extraction/fire_stage1.py \
	src/extraction/fire_stage2_3_single_model.py \
	src/verification/fire_stage4.py \
	src/verification/fire_stage5.py \
	src/state_space/fire_worker2.py \
	src/algebra/fire_worker3_general.py \
	src/instruments/fire_worker3_ace.py \
	src/instruments/fire_worker4_ace.py \
	src/instruments/fire_reconcile_ace.py \
	src/baselines/fire_llm_baseline.py \
	src/baselines/fire_llm_baseline_self_form_included.py \
	src/baselines/fire_llm_structured_baseline.py \
	src/baselines/fire_llm_knowledge_baseline.py \
	src/baselines/trim_state_space.py \
	src/ablation/fire_ablation.py \
	scripts/run_fire.py \
	scripts/evaluate_judgements.py \
	scripts/make_tables.py

.PHONY: help check test selftest check-prompts prompts worked-example tables clean

help:
	@echo "make check           tests, every --selftest, and the prompt-file check"
	@echo "make test            pytest (or the test files directly if pytest is absent)"
	@echo "make selftest        every script's offline --selftest"
	@echo "make check-prompts   prompts/*.txt match the code"
	@echo "make prompts         regenerate prompts/*.txt from the code"
	@echo "make worked-example  the paper's worked example, Worker 3 to the score, in build/"
	@echo "make tables          the paper's tables from the published counts (Markdown)"
	@echo "make clean           remove build/ and Python caches"

check: test selftest check-prompts

test:
	@if $(PYTHON) -c "import pytest" 2>/dev/null; then \
		$(OFFLINE) $(PYTHON) -m pytest -q tests; \
	else \
		$(OFFLINE) $(PYTHON) tests/test_worked_example_chain.py && \
		$(OFFLINE) $(PYTHON) tests/test_scripts.py; \
	fi

selftest:
	@for s in $(SELFTESTS); do \
		echo "== $$s"; \
		$(OFFLINE) $(PYTHON) $$s --selftest > /dev/null || { echo "FAILED: $$s"; exit 1; }; \
	done; echo "all self-tests passed"

check-prompts:
	$(PYTHON) prompts/export_prompts.py --check

prompts:
	$(PYTHON) prompts/export_prompts.py

worked-example:
	mkdir -p build/worked_example
	cp data/worked_example/worker2_output.json build/worked_example/final_mapped_events.json
	$(OFFLINE) $(PYTHON) scripts/run_fire.py --patient-id worked_example --workdir build/worked_example \
		--from worker3 --offline-item-check

tables:
	$(PYTHON) scripts/make_tables.py --published --table all --format markdown

clean:
	rm -rf build .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
