# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr

from numpy import random


def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

arr_pos = []
arr_col = []
arr_tri = []

def get_coord(theta, center, radius):
    return np.array([np.cos(theta) * radius + center[0], np.sin(theta) * radius + center[1]])

def save_img(img, imgname):
    img = img.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
    print("Saving to {}.".format(imgname))
    imageio.imsave(imgname, img)

num_triangles = 100
boundary = 0.8
for i in range(num_triangles):
    center = (random.uniform(-boundary, boundary), random.uniform(-boundary, boundary))
    radius = random.uniform(0.05, 0.07)
    theta = random.uniform(0, np.pi)
    for j in range(3):
        coord = get_coord(theta + j * 2.0 * np.pi / 3.0, center, radius)
        arr_pos.append([coord[0], coord[1], 0, 1])
        arr_col.append([0.5, 0.5, 0.5])
    arr_tri.append([3 * i, 3 * i + 1, 3 * i + 2])

pos = tensor([arr_pos], dtype=torch.float32)
col = tensor([arr_col], dtype=torch.float32)
tri = tensor(arr_tri, dtype=torch.int32)

glctx = dr.RasterizeGLContext()
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
out, _ = dr.interpolate(col, rast, tri)

img = dr.antialias(out, rast, pos, tri)

save_img(img, "tri.png")
save_img(out, "out.png")

