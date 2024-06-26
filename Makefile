.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install .
	.venv/bin/pip install .[dev]


frames:
	mkdir -p frames

test:
	srun --mem 32G --account extremedata --time 1:00:00\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 512 --euler_plotting True --svd_outfile euler_512_svd.h5 --checkpoint_frequency 20 --checkpoint_outfile euler_512_checkpoints.h5


test_gpu:
	srun --mem 32G --gres gpu --time 1:00:00\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 512 --euler_plotting True --svd_outfile euler_512_svd.h5 --checkpoint_frequency 20 --checkpoint_outfile euler_512_checkpoints.h5


test_gpu_1024: | frames
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:40:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n 1024 --euler_plotting True --svd_outfile euler_1024_svd.h5 --checkpoint_frequency 50 --checkpoint_outfile euler_1024_checkpoints.h5 --euler_chunk_size 100 --svd_max_rank 200


test_gpu_1024_noplotting:
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:10:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n=1024 --euler_plotting=False --svd_outfile=noplotting.h5 --checkpoint_frequency=10 -euler_chunk_size=100 --svd_max_rank=200



NGRID = 1024
SLURMPARAMS = --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00 --gres=gpu:h100:1
SWEEP=--grid_n=1024 --euler_plotting=True --compute_svd=False --euler_chunk_size=100
RUNCMD = .venv/bin/python sqm_demo/tracked_euler.py 

offset_x_sweep: | frames
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.400 --frame_basename=0.400 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.450 --frame_basename=0.450 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.490 --frame_basename=0.490 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.500 --frame_basename=0.500 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.510 --frame_basename=0.510 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.550 --frame_basename=0.550 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_offset=0.600 --frame_basename=0.600 &


spread_x_sweep: | frames
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_spread_factor=0.490 --frame_basename=x_spread_0.490 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_spread_factor=0.495 --frame_basename=x_spread_0.495 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_spread_factor=0.500 --frame_basename=x_spread_0.500 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_spread_factor=0.505 --frame_basename=x_spread_0.505 &
	srun $(SLURMPARAMS) $(RUNCMD) $(SWEEP) --velocity_x_spread_factor=0.510 --frame_basename=x_spread_0.510 &


test_gpu_param: | frames
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.4
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.6
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.5
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.55
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.45
	# srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
	# 	 --gres=gpu:h100:1\
	# 	.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.505
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_x_spread_factor=0.495


test_gpu_param_y: | frames
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=0.9
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=1.1
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=1.0
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=0.95
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=1.05
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=0.995
	srun --mem 32G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 0:20:00\
		 --gres=gpu:h100:1\
		.venv/bin/python sqm_demo/tracked_euler.py --grid_n ${NGRID} --euler_plotting True --checkpoint_frequency 50 --euler_chunk_size=100 --compute_svd=False --velocity_y_spread_factor=1.005
