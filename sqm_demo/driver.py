from absl import flags, app, logging
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import incremental_svd as isvd
import h5py
import gc
import time
import sqm_demo.euler_flux as efl

jax.config.update('jax_debug_nans', True)
jax.config.update('jax_enable_x64', True)

GRID_N = flags.DEFINE_integer(
    'grid_n',
    1024,
    'The number of grid points per axis.',
)
CFL = flags.DEFINE_float(
    'cfl',
    0.2,
    'The CFL number.',
)
CHUNK_SIZE = flags.DEFINE_integer(
    'euler_chunk_size',
    100,
    'The chunks to simulate',
)
EULER_T_FINAL = flags.DEFINE_float(
    'euler_t_final',
    3.0,
    'The final time of the simulation.',
)

# initial condition parameters
VELOCITY_X_SPREAD_FACTOR = flags.DEFINE_multi_float(
    'velocity_x_spread_factor',
    0.5,
    'The spread factor.',
)
VELOCITY_X_OFFSET = flags.DEFINE_float(
    'velocity_x_offset',
    0.5,
    'The velocity_x offset.',
)
VELOCITY_Y_SPREAD_FACTOR = flags.DEFINE_float(
    'velocity_y_spread_factor',
    0.1,
    'The spread factor.',
)

# plotting parameters
PLOTTING = flags.DEFINE_bool(
    'density_plots',
    False,
    'Whether to generate density plots',
)
FRAME_BASENAME = flags.DEFINE_string(
    'frame_basename',
    'frame',
    'The basename of the frame.',
)

# incremental svd parameters
COMPUTE_SVD = flags.DEFINE_bool(
    'compute_svd',
    True,
    'Whether to develop the incremental svd.',
)
SVD_OUTFILE = flags.DEFINE_string(
    'svd_outfile',
    'test.h5',
    'Where to store the incremental svd.',
)
PATH_STEM = flags.DEFINE_string(
    'path_stem',
    '',
    'Where to store the svd checkpoints',
)

# checkpointing parameters
CHECKPOINT_FREQUENCY = flags.DEFINE_integer(
    'checkpoint_frequency',
    10,
    '',
)
CHECKPOINT_OUTFILE = flags.DEFINE_string(
    'checkpoint_outfile',
    'checkpoints',
    'Where to store the checkpoints.',
)
STORE_CHECKPOINTS = flags.DEFINE_bool(
    'store_checkpoints',
    True,
    'Whether to store checkpoints',
)
TRIGGER_GC_KESTREL = flags.DEFINE_boolean(
    'trigger_gc_kestrel',
    True,
    'Whether to trigger GC and wait for kestrel'
    'to move data from memory to storage',
)


def grid():
  N = GRID_N.value
  dx = 2 / N
  nx = N
  x1 = jnp.linspace(dx / 2 - 1, 1 - dx / 2, nx)
  x2 = jnp.linspace(dx / 2 - 1, 1 - dx / 2, nx)
  Y, X = jnp.meshgrid(x1, x2)
  return nx, dx, X, Y


def hat_profile(x, mag):
  return jnp.tanh(mag * (x + 0.5)) - jnp.tanh(mag * (x - 0.5))


def initialize(x_spread, X, Y, dx):
  c_density = 80
  c_pressure = 80
  B_density = hat_profile(Y, c_density)
  B_vx = hat_profile(Y, c_pressure)
  density = 0.5 + 0.75 * B_density
  pressure = 1 * jnp.ones(X.shape)
  velocity_x = x_spread * (B_vx - 1) + VELOCITY_X_OFFSET.value
  velocity_y = VELOCITY_Y_SPREAD_FACTOR.value * jnp.sin(2 * jnp.pi * X)
  prim_state = jnp.vstack((density, velocity_x, velocity_y, pressure))
  state = efl.conserved_variables(prim_state)
  return state, prim_state, dx


def main(_):
  end_time = EULER_T_FINAL.value
  nx, dx, X, Y = grid()
  svd_state, n_start = isvd.load_or_create_initial_svd(nx**2 * 4)
  fig, ax = plt.subplots()
  if PLOTTING.value:
    img = ax.imshow(
        jnp.zeros((GRID_N.value, GRID_N.value)),
        animated=True,
        vmin=0.6,
        vmax=2.1)
    _ = fig.colorbar(img, ax=ax)
    plt.pause(0.5)
    plt.grid(False)

  logging.info(jax.devices())

  for i_param, vel_x_spread in enumerate(VELOCITY_X_SPREAD_FACTOR.value):
    logging.info("Run simulation with velocity x spread: %s", vel_x_spread)
    state, prim_state, dx = initialize(vel_x_spread, X, Y, dx)
    checkpoints = []
    checkpoint_times = []

    t = 0
    i = 0
    while t < end_time:
      if COMPUTE_SVD.value:
        state, prim_state, t, svd_state = efl.nsteps(
            state,
            prim_state,
            t,
            dx,
            CFL.value,
            svd_state,
        )
      else:
        state, prim_state, t, svd_state = efl.nsteps_fori(
            state,
            prim_state,
            t,
            dx,
            CFL.value,
            svd_state,
        )
      logging.info(
          'Step: %d, Timesteps: %d, Simulated time: %.2e',
          i + 1,
          CHUNK_SIZE.value * (i + 1),
          t,
      )
      logging.info(
          'Datasize: %.6e GB',
          (n_start + svd_state.V.shape[0]) * nx**2 * 4 * 8 * 1e-9,
      )
      logging.info(
          'SVD size: %.6e GB',
          (n_start + svd_state.V.shape[0] + nx**2 * 4) *
          flags.FLAGS.svd_max_rank * 8 * 1e-9,
      )
      logging.info(
          'GPU-SVD size: %.6e GB',
          (svd_state.V.shape[0] + nx**2 * 4) * flags.FLAGS.svd_max_rank * 8 *
          1e-9,
      )
      if PLOTTING.value:
        img.set_array(
            jnp.minimum(
                2.1, jnp.maximum(0.5,
                                 prim_state[:GRID_N.value, :GRID_N.value].T)))
        img.autoscale()
        plt.pause(0.1)
        plt.savefig(f'frames/{FRAME_BASENAME.value}_{i:03d}.png')
      if STORE_CHECKPOINTS.value:
        if i % CHECKPOINT_FREQUENCY.value == 0:
          checkpoints.append(prim_state)
          checkpoint_times.append(t)
      i = i + 1

    if TRIGGER_GC_KESTREL.value:
      logging.info(
          'Triggering GC and waiting for kestrel to remove data from memory',)
      del state
      del prim_state
      time.sleep(5)
      gc.collect()
      time.sleep(5)
      gc.collect()

    logging.info(f'Simulation #{i_param} complete')
    if STORE_CHECKPOINTS.value:
      logging.info('Storing checkpoints')
      with h5py.File(
          f"{PATH_STEM.value}checkpoints/{CHECKPOINT_OUTFILE.value}_{i_param}.h5",
          "w") as f:
        checkpoints = jnp.stack(checkpoints)
        checkpoints = jnp.reshape(checkpoints, (checkpoints.shape[0], -1))
        f.create_dataset("data", data=checkpoints.T)
        f.create_dataset("times", data=jnp.stack(checkpoint_times))

    if PLOTTING.value:
      plt.savefig(f'frames/{FRAME_BASENAME.value}_final.png')

  if COMPUTE_SVD.value:
    logging.info('Storing svd')
    isvd.store_svd(
        svd_state,
        jnp.zeros(nx**2 * 4),
        f"{PATH_STEM.value}svd_files/{SVD_OUTFILE.value}.h5",
    )


if __name__ == "__main__":
  app.run(main)
