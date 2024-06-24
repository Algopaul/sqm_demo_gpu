from absl import flags, app, logging
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import incsvdpack.isvd_base as isvd
import h5py
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_enable_x64', True)

GAS_GAMMA = flags.DEFINE_float('gas_gamma', 1.4, 'The gas constant.')
GRID_N = flags.DEFINE_integer('grid_n', 1024, 'The number of grid points per axis.')
CFL = flags.DEFINE_float('cfl', 0.3, 'The CFL number.')
CHUNK_SIZE = flags.DEFINE_integer('euler_chunk_size', 100, 'The chunks to simulate')
PLOTTING = flags.DEFINE_bool('euler_plotting', False, 'Whether to generate density plots')
EULER_SVD_MAXRANK = flags.DEFINE_integer('euler_svd_max_rank', 200, 'The maximal rank of the svd.')
EULER_T_FINAL = flags.DEFINE_float('euler_t_final', 3.0, 'The final time of the simulation.')
COMPUTE_SVD = flags.DEFINE_bool('compute_svd', True, 'Whether to develop the incremental svd.')
SVD_OUTFILE = flags.DEFINE_string('svd_outfile', 'test.h5', 'Where to store the incremental svd.')
CHECKPOINT_FREQUENCY = flags.DEFINE_integer('checkpoint_frequency', 10, '')
CHECKPOINT_OUTFILE = flags.DEFINE_string('checkpoint_outfile', 'simulation_data.h5', '')


def conserved_variables(primitive_variables):
  density, velocity_x, velocity_y, pressure = jnp.split(primitive_variables, 4)
  momentum_x = density*velocity_x
  momentum_y = density*velocity_y
  e = pressure/((GAS_GAMMA.value-1)*density)
  energy = density*(0.5*(velocity_x**2+velocity_y**2)+e)
  return jnp.vstack((density, momentum_x, momentum_y, energy))


def primitive_variables(conserved_variables):
  density, momentum_x, momentum_y, energy = jnp.split(conserved_variables, 4)
  velocity_x = momentum_x/density
  velocity_y = momentum_y/density
  pressure = (energy-0.5*(velocity_x**2+velocity_y**2)*density)*(GAS_GAMMA.value-1)
  return jnp.vstack((density, velocity_x, velocity_y, pressure))


def vstack4(vec):
  return jnp.vstack((vec, vec, vec, vec))


def euler_flux_x(conserved_state, primitive_state):
  density, _, _, energy = jnp.split(conserved_state, 4)
  _, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  flux_x = jnp.vstack((
    density*velocity_x,
    density*velocity_x**2+pressure,
    density*velocity_x*velocity_y,
    velocity_x*(energy+pressure)
    ))
  return flux_x


def euler_flux_y(conserved_state, primitive_state):
  density, _, _, energy = jnp.split(conserved_state, 4)
  _, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  flux_y = jnp.vstack((
    density*velocity_y,
    density*velocity_x*velocity_y,
    density*velocity_y**2+pressure,
    velocity_y*(energy+pressure)
    ))
  return flux_y


def roll_state(state, idx, axis): # periodic boundary conditions
  x0, x1, x2, x3 = jnp.split(state, 4)
  x0s = jnp.roll(x0, idx, axis)
  x1s = jnp.roll(x1, idx, axis)
  x2s = jnp.roll(x2, idx, axis)
  x3s = jnp.roll(x3, idx, axis)
  return jnp.vstack((x0s, x1s, x2s, x3s))


def hll_flux_x(conservative_L, conservative_R, primitive_L, primitive_R):
  flow_eigvals_L = flow_eigvals_x(conservative_L, primitive_L)
  flow_eigvals_R = flow_eigvals_x(conservative_R, primitive_R)
  sminp = jnp.min(flow_eigvals_L, axis=0)
  smaxp = jnp.max(flow_eigvals_R, axis=0)
  smin = vstack4(sminp)
  smax = vstack4(smaxp)
  f_L = euler_flux_x(conservative_L, primitive_L)
  f_R = euler_flux_x(conservative_R, primitive_R)
  conservative_avg = (smax*conservative_R-smin*conservative_L+f_L-f_R)/(smax-smin)
  primitive_avg = primitive_variables(conservative_avg)
  flow_eigvals_avg = flow_eigvals_x(conservative_avg, primitive_avg)
  smin = vstack4(jnp.min(jnp.stack((*flow_eigvals_avg, sminp)), axis=0))
  smax = vstack4(jnp.max(jnp.stack((*flow_eigvals_avg, smaxp)), axis=0))
  f_avg = (smax*f_L-smin*f_R+smax*smin*(conservative_R-conservative_L))/(smax-smin)
  return (smin>0)*f_L+(smin<=0)*(smax>=0)*f_avg+(smax<0)*f_R


def hll_flux_y(conservative_L, conservative_R=None, primitive_L=None, primitive_R=None):
  if conservative_R is None:
    conservative_R = roll_state(conservative_L, -1, axis=1)
    primitive_L = primitive_variables(conservative_L)
    primitive_R = primitive_variables(conservative_R)
  flow_eigvals_L = flow_eigvals_y(conservative_L, primitive_L)
  flow_eigvals_R = flow_eigvals_y(conservative_R, primitive_R)
  sminp = jnp.min(flow_eigvals_L, axis=0)
  smaxp = jnp.max(flow_eigvals_R, axis=0)
  smin = vstack4(sminp)
  smax = vstack4(smaxp)
  f_L = euler_flux_y(conservative_L, primitive_L)
  f_R = euler_flux_y(conservative_R, primitive_R)
  conservative_avg = (smax*conservative_R-smin*conservative_L+f_L-f_R)/(smax-smin)
  primitive_avg = primitive_variables(conservative_avg)
  flow_eigvals_avg = flow_eigvals_y(conservative_avg, primitive_avg)
  smin = vstack4(jnp.min(jnp.stack((*flow_eigvals_avg, sminp)), axis=0))
  smax = vstack4(jnp.max(jnp.stack((*flow_eigvals_avg, smaxp)), axis=0))
  f_avg = (smax*f_L-smin*f_R+smax*smin*(conservative_R-conservative_L))/(smax-smin)
  return (smin>0)*f_L+(smin<=0)*(smax>=0)*f_avg+(smax<0)*f_R


def flow_eigvals_x(conservative_state, primitive_state):
  density, _, _, _ = jnp.split(conservative_state, 4)
  _, velocity_x, _, pressure = jnp.split(primitive_state, 4)
  # enthalpy = (energy + pressure)/density
  # sound_speed = jnp.sqrt((GAS_GAMMA.value-1)*(enthalpy-0.5*velocity_x**2))
  sound_speed = jnp.sqrt(GAS_GAMMA.value*pressure/density)
  lambda_x = jnp.stack((velocity_x-sound_speed, velocity_x, velocity_x+sound_speed))
  return lambda_x


def flow_eigvals_y(conservative_state, primitive_state):
  density, _, _, _ = jnp.split(conservative_state, 4)
  _, _, velocity_y, pressure = jnp.split(primitive_state, 4)
  # enthalpy = (energy + pressure)/density
  # sound_speed = jnp.sqrt((GAS_GAMMA-1)*(enthalpy-0.5*velocity_y**2))
  sound_speed = jnp.sqrt(GAS_GAMMA.value*pressure/density)
  lambda_y = jnp.stack((velocity_y-sound_speed, velocity_y, velocity_y+sound_speed))
  return lambda_y



def central_differences(v, axis, dx, omega=0.5):
  dL = (v-roll_state(v, 1, axis))/dx
  dR = (roll_state(v, -1, axis)-v)/dx
  return omega*dL+(1-omega)*dR


def first_order_approx(v, v_mean, dx, v_dx=None, v_dy=None):
  if v_dx is None:
    v_dx = central_differences(v, 0, dx)
  if v_dy is None:
    v_dy = central_differences(v, 1, dx)
  v_xL = v_mean + v_dx*dx/2
  v_xR = roll_state(v_mean - v_dx*dx/2, -1, axis=0)
  v_yL = v_mean + v_dy*dx/2
  v_yR = roll_state(v_mean - v_dy*dx/2, -1, axis=1)
  return v_xL, v_xR, v_yL, v_yR



@jax.jit
def full_step(conservative_state, dt, dx):
  a=dt/dx
  c=conservative_state
  p=primitive_variables(c)
  c_Lx, c_Rx, c_Ly, c_Ry = first_order_approx(c, c, dx)
  p_Lx, p_Rx, p_Ly, p_Ry = first_order_approx(p, p, dx)
  Fp_x = hll_flux_x(c_Lx, c_Rx, p_Lx, p_Rx)
  Fm_x = roll_state(Fp_x, 1, 0)
  Fp_y = hll_flux_y(c_Ly, c_Ry, p_Ly, p_Ry)
  Fm_y = roll_state(Fp_y, 1, 1)
  fdiff = (Fp_x-Fm_x)+(Fp_y-Fm_y)
  return conservative_state - a * fdiff


def complete_step(conservative_state, primitive_state, t, dx, CFL):
  density, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  a = jnp.sqrt(GAS_GAMMA.value*pressure/density)
  max_velocity_x = jnp.max(jnp.abs(velocity_x)+a)
  max_velocity_y = jnp.max(jnp.abs(velocity_y)+a)
  dt = CFL*dx/jnp.sqrt(2)/jnp.sqrt(max_velocity_x**2+max_velocity_y**2)
  conservative_state=full_step(conservative_state, dt, dx)
  primitive_state=primitive_variables(conservative_state)
  return conservative_state, primitive_state, dt+t


def nsteps(state, prim_state, t, n, dx, CFL, svd_state):

  def body_fun(col, _):
    state, prim_state, t = col
    col_new = complete_step(state, prim_state, t, dx, CFL)
    return col_new, col_new

  logging.info('Start  %i steps integration at t=%.2e', n, t)
  _, (state, prim_state, t) = jax.lax.scan(body_fun, (state, prim_state, t), jnp.arange(n))
  logging.info('Finish %i steps integration to t=%.2e', n, t[-1])
  if COMPUTE_SVD.value:
    prim_state_flat = jnp.reshape(prim_state, (n, -1)).T
    logging.info('Start incremental svd update with new chunk size %d, %d', *prim_state_flat.shape)
    svd_state = isvd.svd_new_chunk(svd_state, prim_state_flat)
    svd_state = isvd.truncate_svd_max_rank(svd_state, EULER_SVD_MAXRANK.value)
    logging.info('Incremental svd update complete')
  return state[-1], prim_state[-1], t[-1], svd_state


def main(_):
  N = GRID_N.value
  dx=2/N
  nx = N + 1
  x1 = jnp.linspace(dx/2-1, 1-0.5*dx, nx)
  x2 = jnp.linspace(dx/2-1, 1-0.5*dx, nx)
  Y, X = jnp.meshgrid(x1, x2)
  T = EULER_T_FINAL.value
  c_density = 80
  c_pressure = 80
  B_density = jnp.tanh(c_density*Y + c_density/2)-jnp.tanh(c_density*Y-c_density/2)
  B_vx = jnp.tanh(c_pressure*Y + c_pressure/2)-jnp.tanh(c_pressure*Y-c_pressure/2)
  density =  0.5+0.75*B_density
  pressure = 1 * jnp.ones(X.shape)
  velocity_x = 0.5*(B_vx-1) + 0.5
  velocity_y = 0.1*jnp.sin(2*jnp.pi*X)
  prim_state=jnp.vstack((density, velocity_x, velocity_y, pressure))
  state = conserved_variables(prim_state)
  fig, ax = plt.subplots()
  img = ax.imshow(jnp.zeros((N, N)), animated=True, vmin=0.6, vmax=2.1)
  _ = fig.colorbar(img, ax=ax)
  plt.pause(0.5)
  plt.grid(False)
  NJAX = CHUNK_SIZE.value
  svd_state = isvd.initialize_empty_svd(nx**2*4)
  checkpoints = []
  checkpoint_times = []

  t=0
  for i in range(10000):
    state, prim_state, t, svd_state = nsteps(state, prim_state, t, NJAX, dx, CFL.value, svd_state)
    logging.info('Step: %d, Timesteps: %d, Simulated time: %.2e', i+1, NJAX*(i+1), t)
    logging.info('Datasize: %.6e GB', NJAX*(i+1)*nx**2*4*8*1e-9)
    logging.info('SVD size: %.6e GB', (NJAX*(i+1)+nx**2*4)*EULER_SVD_MAXRANK.value*8*1e-9 )
    if PLOTTING.value:
      img.set_array(jnp.minimum(2.1, jnp.maximum(0.5, prim_state[:N, :N].T)))
      img.autoscale()
      plt.pause(0.1)
      plt.savefig(f'frame_{N:04d}_{i:03d}.png')
    if t > T:
      break
    if i % CHECKPOINT_FREQUENCY.value == 0:
      checkpoints.append(prim_state)
      checkpoint_times.append(t)


  isvd.store_svd(svd_state, jnp.zeros(nx**2*4), SVD_OUTFILE.value)
  with h5py.File(CHECKPOINT_OUTFILE.value, "w") as f:
    checkpoints = jnp.stack(checkpoints)
    checkpoints = jnp.reshape(checkpoints, (checkpoints.shape[0], -1))
    f.create_dataset("data", data=checkpoints.T)
    f.create_dataset("times", data=jnp.stack(checkpoint_times))
  

  if PLOTTING.value:
    plt.savefig(f'final_pic_{N}.png')


if __name__ == "__main__":
  app.run(main)

