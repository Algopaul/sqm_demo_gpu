"""Component functions for computing incremental SVDS"""
import h5py
import jax.numpy as np
import jax
from collections import namedtuple
from absl import logging, flags

SVD_MAXRANK = flags.DEFINE_integer('svd_max_rank', 200, 'The maximal rank of the svd.')
SVD_AUTOTRUNC = flags.DEFINE_bool('svd_autotrunc', True, 'Use autotrunc formulation.')
SVD_INITIAL_FILE = flags.DEFINE_string('svd_initial_file', '', 'Load initial svd from here')


IncrementalSVDState = namedtuple("IncrementalSVDState", "U S V")



def load_or_create_initial_svd(state_dim):
  if SVD_INITIAL_FILE.value == '':
    return initialize_empty_svd(state_dim), 0
  else:
    logging.info("Loading initial svd.")
    return load_svd(SVD_INITIAL_FILE.value, set_V_id=True)


def load_svd(filename, set_V_id=False):
  with h5py.File(filename, 'r') as f:
    U = np.asarray(f['U'])
    S = np.asarray(f['S'])
    if set_V_id:
      V = np.eye(S.shape[0])
      nn=f['V'].shape[1]-S.shape[0] # pyright: ignore
    else:
      V = np.asarray(f['V'])
      nn = V.shape[1]
  return IncrementalSVDState(U, S, V), nn




def initialize_incremental_svd(M):
  U, S, VT = np.linalg.svd(M, full_matrices=False)
  return IncrementalSVDState(U, S, VT.T)


def initialize_empty_svd(n_rows):
  U = np.zeros((n_rows, SVD_MAXRANK.value))
  S = np.zeros(SVD_MAXRANK.value)
  V = np.zeros((0, SVD_MAXRANK.value))
  iss = IncrementalSVDState(U, S, V)
  return iss


def dstack(A):
  A0, A1 = A
  Z0 = np.zeros((A0.shape[0], A1.shape[1]))
  Z1 = np.zeros((A1.shape[0], A0.shape[1]))
  return np.vstack((np.hstack((A0, Z0)), np.hstack((Z1, A1))))


def dstack_eye(A, k):
  Id = np.eye(k)
  return dstack((A, Id))


def qr_update(Q, R, B):
  qmB = Q.T @ B
  BmqB = B-Q @ qmB
  QB, RB = np.linalg.qr(BmqB, 'reduced')
  R = np.hstack((R, qmB))
  R = np.vstack((R, np.hstack((np.zeros((RB.shape[0], Q.shape[1])), RB))))
  return np.hstack((Q, QB)), R


def qr_update_full(Q, R, B):
  A=np.hstack((Q * R, B))
  return np.linalg.qr(A)


@jax.jit
def svd_new_chunk_autotruncate(
    iss: IncrementalSVDState,
    new_chunk,
    ):
  Q, R = qr_update_full(iss.U, iss.S, new_chunk)
  Ui, Si, ViT = np.linalg.svd(R, full_matrices=False)
  S = Si[:SVD_MAXRANK.value]
  U = Q @ Ui[:, :SVD_MAXRANK.value]
  V = np.vstack((iss.V @ ViT.T[:iss.V.shape[1], :SVD_MAXRANK.value], ViT.T[iss.V.shape[1]:, :SVD_MAXRANK.value]))
  return IncrementalSVDState(U, S, V)


def update_and_truncate(iss: IncrementalSVDState, new_chunk):
  if (len(iss.S) >= SVD_MAXRANK.value) and (SVD_AUTOTRUNC.value is True):
    logging.info("Auto-truncate")
    return svd_new_chunk_autotruncate(iss, new_chunk)
  else:
    iss = svd_new_chunk(iss, new_chunk)
    iss = truncate_svd_max_rank(iss, SVD_MAXRANK.value)
  return iss



@jax.jit
def svd_new_chunk(
    iss: IncrementalSVDState,
    new_chunk,
):
  chunk_size = new_chunk.shape[1]
  # Q, R = qr_update(iss.U, np.diag(iss.S), new_chunk)
  Q, R = qr_update_full(iss.U, iss.S, new_chunk)

  W = dstack_eye(iss.V, chunk_size)
  Ui, Si, ViT = np.linalg.svd(R, full_matrices=False)
  S = Si
  U = Q @ Ui
  V = W @ ViT.T
  return IncrementalSVDState(U, S, V)


def truncate_svd_max_rank(iss: IncrementalSVDState, max_rank: int):
  if len(iss.S) > max_rank:
    U = iss.U[:, :max_rank]
    S = iss.S[:max_rank]
    V = iss.V[:, :max_rank]
    # mu_new = iss.S[max_rank:]
    # mu = [*iss.mu, *mu_new]
    return IncrementalSVDState(U, S, V)
  else:
    return iss


# def update_and_truncate(iss: IncrementalSVDState, new_chunk, max_rank):
#   iss = svd_new_chunk(iss, new_chunk)
#   logging.info("Orthogonality %s", iss.U[:, -1].T @ iss.U[:, 0])
#   return truncate_svd_max_rank(iss, max_rank)


def svd_new_col(iss: IncrementalSVDState, new_col):
  new_chunk = np.expand_dims(new_col, 1)
  return svd_new_chunk(iss, new_chunk)


def store_svd(iss, shift, filename, runtime=-1.0):
  with h5py.File(filename, 'w') as file:
    file.create_dataset('U', data=iss.U)
    file.create_dataset('S', data=iss.S)
    file.create_dataset('V', data=iss.V)
    file.create_dataset('shift', data=shift)
    # file.create_dataset('rejected_svals', data=np.array(iss.mu))
    file.create_dataset('runtime', data=np.array([runtime]))
    file.close()
  pass
