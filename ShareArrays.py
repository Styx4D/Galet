import pickle
import numpy as np
from multiprocessing import shared_memory
import copy

def shareNPArray(var):
    shm = shared_memory.SharedMemory(create=True, size=var.nbytes)
    shared_var = np.ndarray(var.shape, dtype=var.dtype, buffer=shm.buf)
    shared_var[:] = var[:]  # Copy the original data into shared memory
    share_dict = {'Shape':shared_var.shape, 'DType':shared_var.dtype, 'SpaceName':shm.name}
    return shared_var, shm, share_dict

def readSharedNPArray(share_dict):
    shm = shared_memory.SharedMemory(name=share_dict['SpaceName'])
    shared_var = np.ndarray(share_dict['Shape'], dtype=share_dict['DType'], buffer=shm.buf)
    return shared_var, shm

def closeSharedNPArrays(share_list, unlink=False):
    for s in share_list:
        s.close()
        if unlink:
            s.unlink()