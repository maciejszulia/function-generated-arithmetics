PY ?= python

.PHONY: all verify problems test figures papers clean

all: verify problems test figures papers

verify:
	$(PY) scripts/verify.py

problems:
	$(PY) -m src.problems

test:
	$(PY) -m pytest -q

figures: problems
	$(PY) scripts/figures.py

papers:
	cd paper && pdflatex -interaction=nonstopmode generator_calculus_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode generator_calculus_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_pl.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_en.tex >/dev/null
	cd paper && pdflatex -interaction=nonstopmode paper_en.tex >/dev/null
	@echo "paper/paper_pl.pdf paper/paper_en.pdf paper/generator_calculus_pl.pdf"

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__ .pytest_cache
