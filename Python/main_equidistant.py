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
import cv2

import cupy as cp
import open3d as o3d

from vfs import VFS
from vfs_params import vfs_params

from equidistant_cam import EquidistantCam
from vector_field import get_vector_fields
from vector_field import get_grid_zero
from vfs_icra_dataset import dataset_loader
from colormap import depth_to_colormap

loader = dataset_loader('/path/to/icra_dataset')
id_img = 0
near_z, far_z = 1.5, 10.
show_point_cloud = True
regenerate_vector_field = True

# reference camera
p = loader.get_calibration()
cam_ref = EquidistantCam(p)
img_ref, img_other = loader.load_image(id_img)
R, t = loader.load_pose(id_img)
D_ref = loader.load_depth(id_img)
mask = np.full(img_ref.shape[:2], 0, dtype=np.uint8)
cv2.circle(mask, (mask.shape[1]//2, mask.shape[0]//2), mask.shape[1]//2 - 10, 255, -1)
cv2.imwrite('icra_dataset_mask.png', mask)

cam_ref.setup_module()
cam_ref.set_mask(mask)

if regenerate_vector_field:
    normalizer = lambda x: x / (cp.linalg.norm(x, axis=2)[:,:,cp.newaxis])
    tvf, cvf = get_vector_fields(get_grid_zero(img_ref.shape[:2]),
                                    cam_ref, cam_ref,
                                    R, t, 1, normalizer)
else:
    tvf, cvf = loader.load_vector_field(id_img)

equi_ref = cv2.equalizeHist(cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY))
equi_other = cv2.equalizeHist(cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY))

param = vfs_params(equi_ref)
vfs = VFS(param)
vfs.setup_module()

vfs.compute_pyramid_sizes()
vfs.create_mask_pyramid(mask)
vfs.create_vector_field_pyramid(tvf, cvf)
vfs.solve_stereo(equi_ref, equi_other)
xyz = vfs.triangulate_flow(cam_ref, t)

xyz_ref = cam_ref.get_xyz_from_distance(cp.array(D_ref)).get()
invalid_xyz = xyz_ref[:,:,2] < 1E-7
cv2.imwrite('img_ref.png', img_ref)
cv2.imwrite('img_other.png', img_other)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ref.reshape(-1, 3))
pcd.colors = o3d.utility.Vector3dVector(np.fliplr(img_ref.reshape(-1, 3)/255))
o3d.io.write_point_cloud('icra_dataset_{:04d}.pcd'.format(id_img), pcd)
if show_point_cloud:
    o3d.visualization.draw_geometries([pcd])

# write depth color map
Dimg = depth_to_colormap(xyz[:,:,2].get(), near_z, far_z)
cv2.imwrite('depth.png', Dimg)
Dimg_ref = depth_to_colormap(xyz_ref[:,:,2], near_z, far_z)
cv2.imwrite('depth_gt.png', Dimg_ref)
