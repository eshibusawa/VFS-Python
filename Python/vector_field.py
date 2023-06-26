# This file is part of VFS-Python.
# Copyright (c) 2023, Eijiro SHIBUSAWA <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import cupy as cp

def get_vector_fields(grid, camera_ref, camera_other, R, t, t_scale, normalizer):
    if isinstance(t_scale, float) or isinstance(t_scale, int):
        t_scale = (t_scale, t_scale)
    XYZ = camera_ref.get_XYZs(grid)
    XYZ = normalizer(XYZ)

    # translation vector field
    # forward path
    st = t_scale[0] * t
    st = cp.array(st[cp.newaxis, cp.newaxis,:,0], dtype=cp.float32)
    uv_other = camera_ref.get_UVs(XYZ + st)
    tvf_forward = uv_other
    # backward path
    st = t_scale[1] * t
    st = cp.array(st[cp.newaxis, cp.newaxis,:,0], dtype=cp.float32)
    uv_other = camera_ref.get_UVs(XYZ - st)
    tvf_backward = uv_other
    # add and normalize
    tvf = (tvf_forward - tvf_backward) / 2
    norm = cp.linalg.norm(tvf, axis=2)
    tvf /= norm[:,:,cp.newaxis]

    # calibration vector field
    XYZ_ = XYZ.reshape(-1, 3)
    RXYZ = cp.dot(XYZ_, cp.array(R.T, dtype=cp.float32)).reshape(XYZ.shape)
    uv_other = camera_other.get_UVs(RXYZ)
    cvf = uv_other - grid

    return tvf.get(), cvf.get()

def get_grid_zero(sz):
    uv = np.empty((*sz, 2), dtype=cp.int32)
    uv[:,:,0] = np.arange(0, sz[1])[np.newaxis,:]
    uv[:,:,1] = np.arange(0, sz[0])[:,np.newaxis]
    return cp.array(uv, dtype=cp.float32)

def get_grid_one(sz):
    return get_grid_zero(sz) + 1
