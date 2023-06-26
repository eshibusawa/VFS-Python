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

import math
import os

import numpy as np
import cupy as cp

from texture import create_texture_object

class VFS:
    def __init__(self, param):
        self.param = param
        self.gpu_module = None

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'vfs.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        min_normalized_grad = 1E-8
        min_tensor_value = 1E-8
        median_filte_kernel_size = 5
        assert median_filte_kernel_size == 5 or median_filte_kernel_size == 3

        cuda_source = cuda_source.replace('VFS_FSCALE', str(self.param.fScale))
        cuda_source = cuda_source.replace('VFS_BETA', str(self.param.beta))
        cuda_source = cuda_source.replace('VFS_GAMMA', str(self.param.gamma))
        cuda_source = cuda_source.replace('VFS_ALPHA0', str(self.param.alpha0))
        cuda_source = cuda_source.replace('VFS_ALPHA1', str(self.param.alpha1))
        cuda_source = cuda_source.replace('VFS_ETA_P', str(3.0))
        cuda_source = cuda_source.replace('VFS_ETA_Q', str(2.0))
        cuda_source = cuda_source.replace('VFS_LAMBDA', str(self.param.Lambda))
        cuda_source = cuda_source.replace('VFS_MIN_NORMALIZED_GRAD', str(min_normalized_grad))
        cuda_source = cuda_source.replace('VFS_MIN_TESOR_VALUE', str(min_tensor_value))
        cuda_source = cuda_source.replace('VFS_MEDIAN_FILTER_KERNEL_SIZE', str(median_filte_kernel_size))
        cuda_source = cuda_source.replace('VFS_UPPER_LIMIT', str(self.param.limitRange))
        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()

    @staticmethod
    def compute_optical_flow(u, vec):
        assert len(u.shape) == 2
        assert len(vec.shape) == 3
        assert u.shape == vec.shape[:2]
        assert vec.shape[2] == 2
        return u[:,:,None] * vec

    def compute_pyramid_sizes(self):
        sizes = list()
        h, w = self.param.height, self.param.width
        for k in range(self.param.nLevel):
            sizes.append((h, w))
            h = int(h / self.param.fScale)
            w = int(w / self.param.fScale)
        self.sizes = sizes

    def create_image_pyramid_(self, img_f32):
        self.setup_module()
        pyr = [img_f32]
        ct = lambda x: create_texture_object(x, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                        filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                        normalizedCoords=1)
        pyr_to = list() # bindless texture object
        gpu_func = self.gpu_module.get_function('downSamplingImage')
        sz_block = 32, 32
        for sz in self.sizes[1:]:
            d = cp.empty(sz, dtype=cp.float32)
            assert d.flags.c_contiguous
            to = ct(pyr[-1])
            sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block, grid=sz_grid,
                args=(
                    d, to, d.shape[0], d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            pyr.append(d)
            pyr_to.append(to)
        to = ct(pyr[-1])
        pyr_to.append(to)

        return pyr, pyr_to

    def create_image_pyramid(self, img_ref, img_other):
        self.setup_module()

        assert img_ref.shape == img_other.shape
        refs, others = list(), list()

        img_ref_gpu = cp.array(img_ref, dtype=cp.uint8)
        ref0 = cp.empty(img_ref_gpu.shape, dtype=cp.float32)
        assert ref0.flags.c_contiguous
        img_other_gpu = cp.array(img_other, dtype=cp.uint8)
        other0 = cp.empty(img_other_gpu.shape, dtype=cp.float32)
        assert other0.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('castAndScale')
        sz_block = 32, 32
        sz_grid = math.ceil(ref0.shape[1] / sz_block[0]), math.ceil(ref0.shape[0] / sz_block[1])
        for s, d, in zip((img_ref_gpu, img_other_gpu), (ref0, other0)):
            gpu_func(
                block=sz_block, grid=sz_grid,
                args=(
                    d, s, s.shape[0], s.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()

        # reference image
        refs, refs_to = self.create_image_pyramid_(ref0)

        # target image is warped
        to = create_texture_object(other0, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                                filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                                normalizedCoords=1)
        other0_warp = self.warping_image(to, self.calibration)
        others, others_to = self.create_image_pyramid_(other0_warp)

        self.refs, self.others = refs, others
        self.refs_to, self.others_to = refs_to, others_to

    def create_mask_pyramid(self, mask):
        self.setup_module()

        masks = [cp.array(mask, dtype=cp.uint8)]
        masks_to = list() # bindless texture object
        gpu_func = self.gpu_module.get_function('downSamplingMask')
        sz_block = 32, 32
        for sz in self.sizes[1:]:
            d = cp.empty(sz, dtype=cp.uint8)
            assert d.flags.c_contiguous
            to = create_texture_object(masks[-1], addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                        filterMode=cp.cuda.runtime.cudaFilterModePoint,
                                        normalizedCoords=1)
            sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block, grid=sz_grid,
                args=(
                    d, to, d.shape[0], d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            masks.append(d)
            masks_to.append(to)
        to = create_texture_object(masks[-1], addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModePoint,
                                    normalizedCoords=1)
        masks_to.append(to)
        self.masks = masks
        self.masks_to = masks_to

    def create_vector_field_pyramid(self, translation, calibration):
        self.setup_module()
        self.calibration = cp.array(calibration, dtype=cp.float32)

        translations = [cp.array(translation, dtype=cp.float32)]
        translations_to = list()
        gpu_func = self.gpu_module.get_function('downSamplingVectorField')
        sz_block = 32, 32
        for sz in self.sizes[1:]:
            d = cp.empty((*sz, 2), dtype=cp.float32)
            assert d.flags.c_contiguous
            to = create_texture_object(translations[-1], addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                        filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                        normalizedCoords=1)
            sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
            gpu_func(
                block=sz_block, grid=sz_grid,
                args=(
                    d, to, d.shape[0], d.shape[1]
                )
            )
            cp.cuda.runtime.deviceSynchronize()
            translations.append(d)
            translations_to.append(to)
        to = create_texture_object(translations[-1], addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                    normalizedCoords=1)
        translations_to.append(to)
        self.translations = translations
        self.translations_to = translations_to

    def warping_image(self, to, warp):
        self.setup_module()

        gpu_func = self.gpu_module.get_function('warpingImage')
        sz_block = 32, 32
        sz = warp.shape[:2]
        d = cp.empty(sz, dtype=cp.float32)
        assert d.flags.c_contiguous
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, warp, to, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def gaussian(self, to, to_mask, sz):
        self.setup_module()

        gpu_func = self.gpu_module.get_function('gaussianMasked')
        sz_block = 32, 32
        d = cp.empty(sz, dtype=cp.float32)
        assert d.flags.c_contiguous
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, to, to_mask, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_tensor_eta(self, img, to_mask):
        self.setup_module()
        to = create_texture_object(img, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                    normalizedCoords=1)

        gpu_func = self.gpu_module.get_function('computeTensorEtaMasked')
        sz_block = 32, 32
        d_tensor = cp.empty((*img.shape, 3), dtype=cp.float32)
        d_eta = cp.empty((*img.shape, 3), dtype=cp.float32)
        assert d_tensor.flags.c_contiguous
        assert d_eta.flags.c_contiguous
        sz_grid = math.ceil(d_tensor.shape[1] / sz_block[0]), math.ceil(d_tensor.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d_tensor, d_eta, to, to_mask, d_tensor.shape[0], d_tensor.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d_tensor, d_eta

    def find_warping_vector(self, warpUV, to_translation):
        self.setup_module()

        assert warpUV.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('findWarpingVector')
        sz_block = 32, 32
        d = cp.empty_like(warpUV)
        assert d.flags.c_contiguous
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, warpUV, to_translation, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_derivatives(self, wapred_other, to_ref, to_translation):
        self.setup_module()

        gpu_func = self.gpu_module.get_function('computeDerivatives')
        sz_block = 32, 32
        d = cp.empty((*wapred_other.shape, 2), dtype=cp.float32)
        assert d.flags.c_contiguous
        to = create_texture_object(wapred_other, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                    normalizedCoords=1)
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, to_ref, to, to_translation, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def update_dual_variables(self, p, q, sigma, u, v, tensor_abc, to_mask):
        self.setup_module()
        assert p.flags.c_contiguous
        assert q.flags.c_contiguous
        assert u.flags.c_contiguous
        assert v.flags.c_contiguous
        assert tensor_abc.flags.c_contiguous
        assert u.shape == v.shape[:2]

        tp = cp.empty_like(p)
        gpu_func = self.gpu_module.get_function('updateDualVariables')
        sz_block = 32, 32
        sz_grid = math.ceil(p.shape[1] / sz_block[0]), math.ceil(p.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                p, tp, q, cp.float32(sigma), u, v, tensor_abc, to_mask, p.shape[0], p.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return p, tp, q

    def l1_thresholding(self, tau, tp, u, u_, derivatives, eta, to_mask):
        self.setup_module()
        assert tp.flags.c_contiguous
        assert u.flags.c_contiguous
        assert u_.flags.c_contiguous
        assert derivatives.flags.c_contiguous
        assert eta.flags.c_contiguous
        assert u.shape == u_.shape

        us = cp.empty_like(u)
        gpu_func = self.gpu_module.get_function('l1Thresholding')
        sz_block = 32, 32
        sz_grid = math.ceil(us.shape[1] / sz_block[0]), math.ceil(us.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                us, cp.float32(tau), tp, u, u_, derivatives, eta, to_mask, us.shape[0], us.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return us

    def update_primal_variables(self, tau, mu, p, q, u, u_, v_, tensor_abc, eta, to_mask):
        self.setup_module()
        assert p.flags.c_contiguous
        assert q.flags.c_contiguous
        assert u.flags.c_contiguous
        assert u_.flags.c_contiguous
        assert v_.flags.c_contiguous
        assert tensor_abc.flags.c_contiguous
        assert eta.flags.c_contiguous
        assert u_.shape == v_.shape[:2]

        us = cp.empty_like(u_)
        vs = cp.empty_like(v_)
        gpu_func = self.gpu_module.get_function('updatePrimalVariables')
        sz_block = 32, 32
        sz_grid = math.ceil(us.shape[1] / sz_block[0]), math.ceil(us.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                us, vs, cp.float32(tau), cp.float32(mu), p, q, u, u_, v_, tensor_abc, eta, to_mask, us.shape[0], us.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return us, vs

    def median_filter(self, u, to_mask):
        self.setup_module()
        assert u.flags.c_contiguous

        d = cp.empty_like(u)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('medianFilter')
        sz_block = 32, 32
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, u, to_mask, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_optical_flow_with_clamp(self, warpUV_, u_, u, translation_vector, to_mask):
        self.setup_module()
        assert warpUV_.flags.c_contiguous
        assert u_.flags.c_contiguous
        assert u.flags.c_contiguous

        gpu_func = self.gpu_module.get_function('computeOpticalFlowWithClamp')
        sz_block = 32, 32
        sz_grid = math.ceil(warpUV_.shape[1] / sz_block[0]), math.ceil(warpUV_.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                warpUV_, u_, u, translation_vector, to_mask, warpUV_.shape[0], warpUV_.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return warpUV_, u_

    def upsampling_vector_field1(self, u, shape):
        self.setup_module()

        d = cp.empty(shape, dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('upSamplingVectorField1')
        to = create_texture_object(u, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                    normalizedCoords=1)
        sz_block = 32, 32
        sz_grid = math.ceil(shape[1] / sz_block[0]), math.ceil(shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, to, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def upsampling_vector_field(self, warp, shape):
        self.setup_module()

        d = cp.empty((*shape, 2), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('upSamplingVectorField')
        to = create_texture_object(warp, addressMode=cp.cuda.runtime.cudaAddressModeMirror,
                                    filterMode=cp.cuda.runtime.cudaFilterModeLinear,
                                    normalizedCoords=1)
        sz_block = 32, 32
        sz_grid = math.ceil(shape[1] / sz_block[0]), math.ceil(shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, to, d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def solve_stereo(self, img_ref, img_other):
        # pyramid iteration
        self.create_image_pyramid(img_ref, img_other)
        for s, k in zip(self.sizes[::-1], range(len(self.sizes), 0, -1)):
            s2 = *s, 2
            s4 = *s, 4
            if s == self.sizes[-1]:
                warpUV_ref = cp.zeros(s2, dtype=cp.float32) # zero
                u = cp.zeros(s, dtype=cp.float32)
                u_ = cp.zeros(s, dtype=cp.float32)
            assert warpUV_ref.shape == s2
            assert u_.shape == s

            # apply gaussian smoothing and compute anisotropic diffusion tensor
            ref_smoothed = self.gaussian(self.refs_to[k - 1], self.masks_to[k - 1], s)
            abc, eta = self.compute_tensor_eta(ref_smoothed, self.masks_to[k - 1])

            # warping iteration
            for kw in range(self.param.nWarpIters):
                # compute translation vector
                translation_vector_ref = self.find_warping_vector(warpUV_ref, self.translations_to[k - 1])

                # warp other image
                other_warped = self.warping_image(self.others_to[k - 1], warpUV_ref)

                # compute derivatives
                derivatives = self.compute_derivatives(other_warped, self.refs_to[k - 1], self.translations_to[k - 1])

                # initialize variables
                p = cp.zeros(s2, dtype=cp.float32)
                q = cp.zeros(s4, dtype=cp.float32)
                v_ = cp.zeros(s2, dtype=cp.float32)

                # solver iteration
                tau = 1.
                sigma = 1. / tau
                # copy last u
                u_before_solver = cp.copy(u)
                for ks in range(self.param.nSolverIters):
                    if sigma < 1000:
                        mu = 1 / np.sqrt(1 + 0.7 * tau * self.param.timeStepLambda).astype(np.float32)
                    else:
                        mu = 1

                    # update dual variables
                    p, tp, q = self.update_dual_variables(p, q, sigma, u_, v_, abc, self.masks_to[k - 1])

                    # apply thresholding
                    u = self.l1_thresholding(tau, tp, u, u_, derivatives, eta, self.masks_to[k - 1])

                    # update primal variables
                    u_, v_ = self.update_primal_variables(tau, mu, p, q, u, u_, v_, abc, eta, self.masks_to[k - 1])

                    # update
                    sigma = sigma / mu
                    tau = tau * mu

                # post processing
                u_filtered = self.median_filter(u, self.masks_to[k - 1])
                # this function internally overwrites 'warpUV_ref' and 'u_before_solver'
                # 'u' and 'u_before_solver' is identical in the sense of pointer!
                warpUV_ref, u = self.compute_optical_flow_with_clamp(warpUV_ref, u_before_solver, u_filtered, translation_vector_ref, self.masks_to[k - 1])

            if k > 1:
                # get upsampled size
                ss = self.sizes[k - 2]
                u_ = self.upsampling_vector_field1(u, ss)
                u = cp.copy(u_)
                warpUV_ref = self.upsampling_vector_field(warpUV_ref, ss)

        # store optical flow
        warpUV_ref[self.masks[0] == 0,:] = 0
        self.warpUV = warpUV_ref
        return warpUV_ref

    def triangulate_flow(self, camera, translation):
        from vector_field import get_grid_zero
        rays = camera.get_rays(self.warpUV.shape[:2])
        uv = get_grid_zero(self.warpUV.shape[:2])
        rays_other = camera.get_XYZs(uv + self.warpUV)
        assert rays.flags.c_contiguous
        assert rays_other.flags.c_contiguous
        assert rays.dtype == cp.float32
        assert rays_other.dtype == cp.float32
        translation_ = cp.array(translation, dtype=cp.float32)
        assert translation_.dtype == cp.float32

        d = cp.empty_like(rays)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('triangulateRays')
        sz_block = 32, 32
        sz_grid = math.ceil(d.shape[1] / sz_block[0]), math.ceil(d.shape[0] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, rays, rays_other, translation_, self.masks_to[0], d.shape[0], d.shape[1]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d
