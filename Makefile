.venv:
	python3.11 -m venv .venv
	.venv/bin/pip install .
	.venv/bin/pip install .[dev]


frames:
	mkdir -p frames

checkpoints:
	mkdir -p checkpoints

svd_files:
	mkdir -p svd_files

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
SLURMPARAMS = --mem 64G --reservation h100-testing --partition=gpu-h100 --account extremedata --time 7:30:00 --gres=gpu:h100:1
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



# sweeps=$(addprefix --velocity_x_spread_factor=, 0.49 0.51 0.492 0.506 0.494 0.502 0.496 0.498 0.504 0.508)
sweeps=$(addprefix --velocity_x_spread_factor=,0.490 0.491 0.492 0.493 0.494 0.495 0.496 0.497 0.498 0.499 0.501 0.502 0.503 0.504 0.505 0.506 0.507 0.508 0.509 0.510)
test_sweep: | frames checkpoints svd_files
	srun $(SLURMPARAMS) $(RUNCMD) --grid_n=1024 --euler_plotting=False --compute_svd=True --svd_outfile=svd_1024_small_sweep_20 $(sweeps) --checkpoint_outfile=small_sweep_20 --checkpoint_frequency=10


test_petabyte_sweep: | frames checkpoints svd_files
	srun $(SLURMPARAMS) $(RUNCMD) --grid_n=1600 --euler_plotting=False --euler_chunk_size=50 --compute_svd=False --svd_outfile=svd_1024_small_sweep_20 $(sweeps) --checkpoint_outfile=small_sweep_1024_20 --checkpoint_frequency=20


supersmall_sweeps=$(addprefix --velocity_x_spread_factor=,0.4990 0.4991 0.4992 0.4993 0.4994 0.4995 0.4996 0.4997 0.4998 0.4999 0.5001 0.5002 0.5003 0.5004 0.5005 0.5006 0.5007 0.5008 0.5009 0.5010)

test_supersmall_sweep: | frames checkpoints svd_files
	srun $(SLURMPARAMS) $(RUNCMD) --grid_n=1024 --euler_plotting=False --compute_svd=True --svd_outfile=svd_1024_super_small_sweep_20 $(supersmall_sweeps) --checkpoint_outfile=super_small_sweep_20 --checkpoint_frequency=20


reconstruct_from_svd:
	.nogpuvenv/bin/python sqm_demo/reconstruct_snapshot.py --svd_filename=svd_files/svd_1024_small_sweep_0.h5 --state_idx=-1 --state_idx=5000 --outfile rec_0.h5
	.nogpuvenv/bin/python sqm_demo/reconstruct_snapshot.py --svd_filename=svd_files/svd_1024_small_sweep_3.h5 --state_idx=-1 --state_idx=5000 --outfile rec_3.h5


WD=/scratch/ps030/Code/sqm_demo/checkpoints

collect_sweep:
	h5util collect_virtual_dataset --output_files datasweep.h5 --data_fields train_data data --input_files $(addsuffix .h5,$(addprefix $(WD)/small_sweep_, 0.490 0.510 0.492 0.506 0.494 0.502 0.496 0.498 0.504 0.508))
	h5util collect_virtual_dataset --output_files datasweep.h5 --data_fields val_data data --input_files $(addsuffix .h5,$(addprefix $(WD)/small_sweep_, 0.490 0.510 0.492 0.506 0.494 0.502 0.496 0.498 0.504 0.508))
	h5util collect_virtual_dataset --output_files datasweep.h5 --data_fields test_data data --input_files $(addsuffix .h5,$(addprefix $(WD)/small_sweep_, 0.490 0.510 0.492 0.506 0.494 0.502 0.496 0.498 0.504 0.508))


test_trajectory_checkpoints:
	srun $(SLURMPARAMS) $(RUNCMD) --grid_n=1024 --euler_plotting=False --compute_svd=False --svd_outfile=None --velocity_x_spread_factor=0.500 --checkpoint_outfile=test_trajectory --checkpoint_frequency=5



