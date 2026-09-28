# Convenience wrappers. Uses a non-editable install (robust on macOS, where
# uv's editable .pth can get the hidden flag and be skipped by Python's site).
#
#   make install     # one-time: create venv + install deps (non-editable)
#   make reinstall   # after changing source code
#   make config      # show resolved settings
#   make download    # fetch the HotpotQA benchmark
#   make build       # build both indexes (benchmark corpus)
#   make ask Q="..." # ask one question
#   make evaluate    # run the benchmark (RETRIEVAL=dense|bm25|hybrid)
#   make deepeval    # run LLM-as-judge metrics
#   make graph       # inspect Neo4j
#   make neo4j       # start Neo4j via Homebrew
#   make ui          # start the Prefect UI
#
# For your own markdown, put files in data/ then:
#   make build SOURCE=files && make ask Q="..." SOURCE=files

RUN := uv run --no-sync rag-compare
SOURCE ?=
SOURCE_ARG := $(if $(SOURCE),--source $(SOURCE),)
RETRIEVAL ?= dense
Q ?= What is this about?

.PHONY: up down install reinstall fix-pth config download build ask evaluate deepeval graph flow neo4j langfuse langfuse-down ui deps

up:
	bash scripts/start_all.sh

down:
	bash scripts/stop_all.sh

install:
	uv sync --no-editable

reinstall:
	uv sync --no-editable --reinstall-package rag-compare

fix-pth:
	bash scripts/fix_editable_pth.sh

deps:
	uv add $(PKG)

config:
	$(RUN) config

download:
	$(RUN) download $(SOURCE_ARG)

build:
	$(RUN) build $(SOURCE_ARG) --force

ask:
	$(RUN) ask "$(Q)" $(SOURCE_ARG) --retrieval $(RETRIEVAL)

evaluate:
	$(RUN) evaluate $(SOURCE_ARG) --retrieval $(RETRIEVAL)

deepeval:
	$(RUN) deepeval $(SOURCE_ARG) --retrieval $(RETRIEVAL)

graph:
	$(RUN) graph

flow:
	$(RUN) flow all $(SOURCE_ARG)

neo4j:
	brew services start neo4j

langfuse:
	docker compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse UI: http://localhost:3000"

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down

ui:
	uv run --no-sync prefect server start
