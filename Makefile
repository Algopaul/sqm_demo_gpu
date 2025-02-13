include ./config/runner_definition.mk

dirs=data/ data/datafiles frames/ checkpoints/ svd_files/

$(dirs): %:
	mkdir -p ${*}

.venv:
	$(DEFAULTPYTHON) -m venv .venv
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]

install: .venv $(dirs)
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]


small_run: .venv $(dirs)
	$(RUN) .venv/bin/python sqm_demo/driver.py --grid_n 128 --density_plots=True
