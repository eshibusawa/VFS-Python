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

import cupy as cp

def upload_constant(module, arr, key, dtype=cp.float32):
    arr_ptr = module.get_global(key)
    arr_gpu = cp.ndarray(arr.shape, dtype, arr_ptr)
    arr_gpu[:] = cp.array(arr, dtype=dtype)

def download_constant(module, key, shape, dtype=cp.float32):
    arr_ptr = module.get_global(key)
    arr_gpu = cp.ndarray(shape, dtype, arr_ptr)
    return arr_gpu.get()

def get_array_from_ptr(module, ptr, shape, dtype):
    assert (len(shape) == 3) or (len(shape) == 4)
    sz = dtype().itemsize
    mem = cp.cuda.UnownedMemory(ptr, 0, module)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    if len(shape) == 3:
        arr = cp.ndarray(shape=(shape[0], shape[1]), dtype=cp.float32, strides=(sz*shape[2],sz), memptr=memptr)
    else:
        assert (shape[2] == 2) or (shape[2] == 4)
        st = shape[2]
        arr = cp.ndarray(shape=(shape[0], shape[1], shape[2]), dtype=cp.float32, strides=(st*sz*shape[3],st*sz,sz), memptr=memptr)
    return arr
