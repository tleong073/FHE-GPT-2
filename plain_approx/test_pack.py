import numpy as np
import numpy.random as rand
import pack
import pytest

@pytest.mark.pack
def test_pack_from_row():
    A = rand.random((128,768))

    A_packed = pack.pack_from_row(A)
    res = pack.pack_tight(A_packed)

    assert np.equal(res.reshape(-1),A.reshape(-1)).all()

@pytest.mark.pack
def test_pack_heads():
    A = rand.random((12,128,64))

    A_head_packed = pack.pack_heads(A,12,128,64)
    res = pack.unpack_heads(A_head_packed,12,128,64)
    assert np.array_equal(A,res)



