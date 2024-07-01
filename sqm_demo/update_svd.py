"""Update SVD with new psi"""
from absl import logging, flags, app
import numpy as np # using numpy to specifically target CPU
import h5py

SVD_SOURCE_FILE_OLD = flags.DEFINE_string('svd_source_file_old', '', 'The svd to load (h5 file with fields U, S, V)')
SVD_SOURCE_FILE_NEW = flags.DEFINE_string('svd_source_file_new', '', 'The svd to load (h5 file with fields U, S, V)')
SVD_TARGET_FILE = flags.DEFINE_string('svd_target_file', '', 'The svd to produce (h5 file with fields U, S, V=vstack([V_old@PSI, V])')


def main(_):
  logging.info(SVD_SOURCE_FILE_OLD.value)
  logging.info(SVD_SOURCE_FILE_NEW.value)
  logging.info(SVD_TARGET_FILE.value)
  f1=h5py.File(SVD_SOURCE_FILE_OLD.value, 'r')
  f2=h5py.File(SVD_SOURCE_FILE_NEW.value, 'r')
  rank = np.array(f1['S']).shape[0]
  logging.info(f1['V'].shape)
  logging.info(f2['V'].shape)
  V = np.vstack((
    np.array(f1['V']) @ f2['V'][:rank, :],
    f2['V'][rank:, :]
    ))

  with h5py.File(SVD_TARGET_FILE.value, 'w') as f3:
    f3.create_dataset(name='U', data=np.array(f2['U']))
    f3.create_dataset(name='S', data=np.array(f2['S']))
    f3.create_dataset(name='V', data=V)


if __name__=='__main__':
  app.run(main)
