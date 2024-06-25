.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install .[dev]



test:
	srun --mem 32G --time 1:00:00\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 512 --euler_plotting True --svd_outfile euler_512_svd.h5 --checkpoint_frequency 20 --checkpoint_outfile euler_512_checkpoints.h5


test_gpu:
	srun --mem 32G --gres gpu --time 1:00:00\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 512 --euler_plotting True --svd_outfile euler_512_svd.h5 --checkpoint_frequency 20 --checkpoint_outfile euler_512_checkpoints.h5


test_gpu_1024:
	srun --mem 64G --gres gpu:a100 --time 1:00:00\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 1024 --euler_plotting True --svd_outfile euler_1024_svd.h5 --checkpoint_frequency 50 --checkpoint_outfile euler_1024_checkpoints.h5 --euler_chunk_size 50 --euler_svd_max_rank 100
