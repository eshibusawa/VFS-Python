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

import glob
import os
import numpy as np
import cv2

class dataset_loader():
    def __init__(self, base_path):
        self.base_path = base_path
        self.pose_path = os.path.join(base_path, 'calib')
        self.fn_xmls = sorted(glob.glob(os.path.join(self.pose_path, '*.xml')))
        # unit of depth is meter
        # https://github.com/menandro/vfs/issues/11
        self.depth_scale = 256

    def get_calibration(self):
        # equidistant camera model
        # https://gist.github.com/menandro/cd5f4b5309f16f1a0f1987fcb2baf057
        f = 800.0 /(2*np.pi/3)
        cx = 400
        cy = 400

        return f, cx, cy

    def load_pose(self, id):
        fs = cv2.FileStorage(self.fn_xmls[id], cv2.FileStorage_READ)
        # matrix K seems to be ignored
        # https://gist.github.com/menandro/cd5f4b5309f16f1a0f1987fcb2baf057
        R = fs.getNode('R').mat()
        t = fs.getNode('t').mat()

        return R, t

    def load_depth(self, id):
        dn = os.path.join(os.path.join('proj_depth', 'groundtruth'), 'image_02')
        fpfn = self.fn_xmls[id].replace('.xml', '.png').replace('calib', dn)
        depth = cv2.imread(fpfn, cv2.IMREAD_UNCHANGED)
        depth = (depth / self.depth_scale).astype(np.float32)

        return depth

    def load_result(self, id):
        fpfn = self.fn_xmls[id].replace('.xml', 'depth.png').replace('calib', 'result_ours')
        depth = cv2.imread(fpfn, cv2.IMREAD_UNCHANGED)
        depth = (depth / self.depth_scale).astype(np.float32)

        return depth

    def load_image(self, id):
        dn2 = os.path.join('image_02', 'data')
        fpfn2 = self.fn_xmls[id].replace('.xml', '.png').replace('calib', dn2)
        img2 = cv2.imread(fpfn2)

        dn3 = os.path.join('image_03', 'data')
        fpfn3 = self.fn_xmls[id].replace('.xml', '.png').replace('calib', dn3)
        img3 = cv2.imread(fpfn3)

        return img2, img3

    def load_vector_field(self, id):
        fpfn_translation = self.fn_xmls[id].replace('.xml', '.flo').replace('calib', 'translationVector')
        translation_vector = cv2.readOpticalFlow(fpfn_translation)
        fpfn_calibration = self.fn_xmls[id].replace('.xml', '.flo').replace('calib', 'calibrationVector')
        calibration_vector = cv2.readOpticalFlow(fpfn_calibration)

        return translation_vector, calibration_vector
