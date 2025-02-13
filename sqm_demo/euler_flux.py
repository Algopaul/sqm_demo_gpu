import jax
import jax.numpy as jnp
from absl import logging
from absl import flags
import incremental_svd as isvd

GAS_GAMMA = flags.DEFINE_float(
    'gas_gamma',
    1.4,
    'The gas constant.',
)


def conserved_variables(primitive_variables):
  density, velocity_x, velocity_y, pressure = jnp.split(primitive_variables, 4)
  momentum_x = density * velocity_x
  momentum_y = density * velocity_y
  e = pressure / ((GAS_GAMMA.value - 1) * density)
  energy = density * (0.5 * (velocity_x**2 + velocity_y**2) + e)
  return jnp.vstack((density, momentum_x, momentum_y, energy))


def primitive_variables(conserved_variables):
  density, momentum_x, momentum_y, energy = jnp.split(conserved_variables, 4)
  velocity_x = momentum_x / density
  velocity_y = momentum_y / density
  pressure = (energy - 0.5 * (velocity_x**2 + velocity_y**2) * density) * (
      GAS_GAMMA.value - 1)
  return jnp.vstack((density, velocity_x, velocity_y, pressure))


def vstack4(vec):
  return jnp.vstack((vec, vec, vec, vec))


def euler_flux_x(conserved_state, primitive_state):
  density, _, _, energy = jnp.split(conserved_state, 4)
  _, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  flux_x = jnp.vstack(
      (density * velocity_x, density * velocity_x**2 + pressure,
       density * velocity_x * velocity_y, velocity_x * (energy + pressure)))
  return flux_x


def euler_flux_y(conserved_state, primitive_state):
  density, _, _, energy = jnp.split(conserved_state, 4)
  _, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  flux_y = jnp.vstack(
      (density * velocity_y, density * velocity_x * velocity_y,
       density * velocity_y**2 + pressure, velocity_y * (energy + pressure)))
  return flux_y


def roll_state(state, idx, axis):  # periodic boundary conditions
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
  conservative_avg = (smax * conservative_R - smin * conservative_L + f_L -
                      f_R) / (
                          smax - smin)
  primitive_avg = primitive_variables(conservative_avg)
  flow_eigvals_avg = flow_eigvals_x(conservative_avg, primitive_avg)
  smin = vstack4(jnp.min(jnp.stack((*flow_eigvals_avg, sminp)), axis=0))
  smax = vstack4(jnp.max(jnp.stack((*flow_eigvals_avg, smaxp)), axis=0))
  f_avg = (smax * f_L - smin * f_R + smax * smin *
           (conservative_R - conservative_L)) / (
               smax - smin)
  return (smin > 0) * f_L + (smin <= 0) * (smax >= 0) * f_avg + (smax < 0) * f_R


def hll_flux_y(conservative_L,
               conservative_R=None,
               primitive_L=None,
               primitive_R=None):
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
  conservative_avg = (smax * conservative_R - smin * conservative_L + f_L -
                      f_R) / (
                          smax - smin)
  primitive_avg = primitive_variables(conservative_avg)
  flow_eigvals_avg = flow_eigvals_y(conservative_avg, primitive_avg)
  smin = vstack4(jnp.min(jnp.stack((*flow_eigvals_avg, sminp)), axis=0))
  smax = vstack4(jnp.max(jnp.stack((*flow_eigvals_avg, smaxp)), axis=0))
  f_avg = (smax * f_L - smin * f_R + smax * smin *
           (conservative_R - conservative_L)) / (
               smax - smin)
  return (smin > 0) * f_L + (smin <= 0) * (smax >= 0) * f_avg + (smax < 0) * f_R


def flow_eigvals_x(conservative_state, primitive_state):
  density, _, _, _ = jnp.split(conservative_state, 4)
  _, velocity_x, _, pressure = jnp.split(primitive_state, 4)
  sound_speed = jnp.sqrt(GAS_GAMMA.value * pressure / density)
  lambda_x = jnp.stack(
      (velocity_x - sound_speed, velocity_x, velocity_x + sound_speed))
  return lambda_x


def flow_eigvals_y(conservative_state, primitive_state):
  density, _, _, _ = jnp.split(conservative_state, 4)
  _, _, velocity_y, pressure = jnp.split(primitive_state, 4)
  sound_speed = jnp.sqrt(GAS_GAMMA.value * pressure / density)
  lambda_y = jnp.stack(
      (velocity_y - sound_speed, velocity_y, velocity_y + sound_speed))
  return lambda_y


def central_differences(v, axis, dx, omega=0.5):
  dL = (v - roll_state(v, 1, axis)) / dx
  dR = (roll_state(v, -1, axis) - v) / dx
  return omega * dL + (1 - omega) * dR


def first_order_approx(v, v_mean, dx, v_dx=None, v_dy=None):
  if v_dx is None:
    v_dx = central_differences(v, 0, dx)
  if v_dy is None:
    v_dy = central_differences(v, 1, dx)
  v_xL = v_mean + v_dx * dx / 2
  v_xR = roll_state(v_mean - v_dx * dx / 2, -1, axis=0)
  v_yL = v_mean + v_dy * dx / 2
  v_yR = roll_state(v_mean - v_dy * dx / 2, -1, axis=1)
  return v_xL, v_xR, v_yL, v_yR


@jax.jit
def full_step(conservative_state, dt, dx):
  a = dt / dx
  c = conservative_state
  p = primitive_variables(c)
  c_Lx, c_Rx, c_Ly, c_Ry = first_order_approx(c, c, dx)
  p_Lx, p_Rx, p_Ly, p_Ry = first_order_approx(p, p, dx)
  Fp_x = hll_flux_x(c_Lx, c_Rx, p_Lx, p_Rx)
  Fm_x = roll_state(Fp_x, 1, 0)
  Fp_y = hll_flux_y(c_Ly, c_Ry, p_Ly, p_Ry)
  Fm_y = roll_state(Fp_y, 1, 1)
  fdiff = (Fp_x - Fm_x) + (Fp_y - Fm_y)
  return conservative_state - a * fdiff


def complete_step(conservative_state, primitive_state, t, dx, CFL):
  density, velocity_x, velocity_y, pressure = jnp.split(primitive_state, 4)
  a = jnp.sqrt(GAS_GAMMA.value * pressure / density)
  max_velocity_x = jnp.max(jnp.abs(velocity_x) + a)
  max_velocity_y = jnp.max(jnp.abs(velocity_y) + a)
  dt = CFL * dx / jnp.sqrt(2) / jnp.sqrt(max_velocity_x**2 + max_velocity_y**2)
  conservative_state = full_step(conservative_state, dt, dx)
  primitive_state = primitive_variables(conservative_state)
  return conservative_state, primitive_state, dt + t


@jax.jit
def nsteps(state, prim_state, t, dx, CFL, svd_state):

  def body_fun(col, _):
    state, prim_state, t = col
    col_new = complete_step(state, prim_state, t, dx, CFL)
    return col_new, col_new

  _, (state, prim_state, t) = jax.lax.scan(
      body_fun,
      (state, prim_state, t),
      jnp.arange(flags.FLAGS.euler_chunk_size),
  )
  last_state = state[-1]
  last_prim_state = prim_state[-1]
  prim_state_flat = jnp.reshape(prim_state,
                                (flags.FLAGS.euler_chunk_size, -1)).T
  svd_state = isvd.svd_new_chunk_autotruncate(svd_state, prim_state_flat)
  return last_state, last_prim_state, t[-1], svd_state


def nsteps_fori(state, prim_state, t, dx, CFL, svd_state):

  def body_fun(_, col):
    state, prim_state, t = col
    col_new = complete_step(state, prim_state, t, dx, CFL)
    return col_new

  logging.info('Start  %i steps integration at t=%.2e',
               flags.FLAGS.euler_chunk_size, t)
  state, prim_state, t = jax.lax.fori_loop(
      0,
      flags.FLAGS.euler_chunk_size,
      body_fun,
      (state, prim_state, t),
  )
  logging.info('Finish %i steps integration to t=%.2e',
               flags.FLAGS.euler_chunk_size, t)
  last_state = state
  last_prim_state = prim_state
  return last_state, last_prim_state, t, svd_state
