PY ?= python

.PHONY: all verify test papers clean

all: verify test papers

verify:
	$(PY) scripts/verify.py

test:
	$(PY) -m pytest -q

papers:
	cd paper && pdflatex -interaction=nonstopmode paper_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_en.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_en.tex >/dev/null
	@echo "paper/paper_pl.pdf paper/paper_en.pdf"

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__ .pytest_cache
