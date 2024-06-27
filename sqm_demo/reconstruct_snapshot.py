from absl import flags, app, logging
import jax.numpy as jnp
import h5py
from incremental_svd import load_svd
import matplotlib.pyplot as plt
import jax
jax.config.update('jax_enable_x64', True)


SVD_FILENAME = flags.DEFINE_string('svd_filename', '', 'svd to load')
STATE_IDX = flags.DEFINE_multi_integer('state_idx', [0], 'The state to reconstruct')
OUTFILE = flags.DEFINE_string('outfile', 'reconstructed_states', 'Where to store the reconstructed_states')


def extract_density(rec_state):
  n=int(jnp.sqrt(rec_state.shape[0]/4))
  s=jnp.reshape(rec_state, (4*n, n))
  return s[:n, :n]


def main(_):
  logging.info('Loading svd')
  iss = load_svd(SVD_FILENAME.value)
  US = iss.U * iss.S
  densities = []
  for idx in STATE_IDX.value:
    logging.info('Reconstructing %s', idx)
    v = iss.V[idx, :]
    rec_state = US@v.T
    densities.append(extract_density(rec_state))
  densities = jnp.stack(densities)
  with h5py.File(OUTFILE.value, 'w') as f:
    f.create_dataset('data', data=densities)


if __name__=='__main__':
  app.run(main)
