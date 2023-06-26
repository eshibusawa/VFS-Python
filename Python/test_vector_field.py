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

from unittest import TestCase
from nose.tools import ok_

from vfs_icra_dataset import dataset_loader
from equidistant_cam import EquidistantCam
from vector_field import get_vector_fields
from vector_field import get_grid_one

class VFSVectorFieldTestCase(TestCase):
    def setUp(self):
        self.loader = dataset_loader('/path/to/icra_dataset')
        self.id_img = 103
        self.eps = 5E-4

        p = self.loader.get_calibration()
        self.cam_ref = EquidistantCam(p)
        img_ref, img_other = self.loader.load_image(self.id_img)
        self.R, self.t = self.loader.load_pose(self.id_img)
        self.shape = img_ref.shape[:2]
        self.cam_ref.setup_module()

    def tearDown(self):
        pass

    def vector_field_test(self):
        tvf_ref, cvf_ref = self.loader.load_vector_field(self.id_img)
        self.cam_ref.setup_module()
        normalizer = lambda x: x / (cp.linalg.norm(x, axis=2)[:,:,cp.newaxis])
        tvf, cvf = get_vector_fields(get_grid_one(self.shape[:2]),
                                        self.cam_ref, self.cam_ref,
                                        self.R, self.t, (1, 0.01), normalizer)
        # scale = 0.01 is only for reproduction of the dataset and testing
        # see https://github.com/menandro/vfs/issues/13 detail.

        err = np.abs(tvf_ref - tvf)
        ok_(np.max(err) < self.eps)

        err = np.abs(cvf_ref - cvf)
        ok_(np.max(err) < self.eps)
