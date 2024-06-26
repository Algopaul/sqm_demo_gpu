.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install .
	.venv/bin/pip install .[dev]


frames:
	mkdir -p frames

checkpoints:
	mkdir -p checkpoints

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

test_local: | frames checkpoints
	$(RUNCMD) --grid_n=128 --euler_plotting=False --compute_svd=True --svd_outfile=svd_64_zero_init.h5
	$(RUNCMD) --grid_n=128 --euler_plotting=True --compute_svd=True --svd_initial_file=svd_64_zero_init.h5 --svd_outfile=svd_64_init.h5



sweeps=$(addprefix --velocity_x_spread_factor=,0.49 0.50 0.51)

test_sweep: | frames checkpoints
	$(RUNCMD) --grid_n=128 --euler_plotting=False --compute_svd=True --svd_outfile=svd_64_zero_init.h5 $(sweeps)

