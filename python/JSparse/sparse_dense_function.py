import jittor as jt
from jittor import Function




def spmm(
    rows: jt.Var,
    cols: jt.Var,
    vals: jt.Var,
    size: jt.NanoVector,
    mat: jt.Var,
    spmm_mode='scatter',
    is_sorted: bool = False,
    cuda_spmm_alg: int = 1,
) -> jt.Var:
    assert len(rows) == len(cols), "Invalid length"
    assert len(rows) == len(vals), "Invalid length"
    assert vals.dtype == mat.dtype, "dtype mismatch"

    if jt.flags.use_cuda > 1:
        assert jt.has_cuda == 1, "No GPUs available"
        rows = rows.int()
        cols = cols.int()
        ''' 
        TODO: Using the coo_spmm of cuSPARSE on GPU
        result = coo_spmm_int32(
            rows, cols, vals, size[0], size[1], mat, cuda_spmm_alg, is_sorted
        )
        '''
    else:
        if (spmm_mode == 'scatter'):


        
        
        




class SPMM(Function):
