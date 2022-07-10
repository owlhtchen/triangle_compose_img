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
from torch import optim
import cv2

from numpy import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)

arr_pos = []
arr_col = []
arr_tri = []

def get_coord(theta, center, radius):
    return np.array([np.cos(theta) * radius + center[0], np.sin(theta) * radius + center[1]])

def save_img(img, imgname):
    with torch.no_grad():
        img = img.cpu().numpy()[0, :, :, :]
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        print("Saving to {}.".format(imgname))
        imageio.imsave(imgname, img)

num_triangles = 200
boundary = 0.8

xy_coords = torch.zeros([num_triangles * 3, 2], dtype=torch.float32).to(device)
# col_value = torch.zeros([num_triangles * 3, 1], dtype=torch.float32).to(device)

for i in range(num_triangles):
    center = (random.uniform(-boundary, boundary), random.uniform(-boundary, boundary))
    radius = random.uniform(0.05, 0.07)
    theta = random.uniform(0, np.pi)
    for j in range(3):
        coord = get_coord(theta + j * 2.0 * np.pi / 3.0, center, radius)
        xy_coords[3 * i + j, 0], xy_coords[3 * i + j, 1] = coord[0], coord[1]
        arr_col.append([0.5, 0.5, 0.5])
        # col_value[3 * i + j, 0] = 0.5
    arr_tri.append([3 * i, 3 * i + 1, 3 * i + 2])

col = tensor([arr_col], dtype=torch.float32).requires_grad_().to(device)
tri = tensor(arr_tri, dtype=torch.int32)
xy_coords.requires_grad_()
# col_value.requires_grad_()

glctx = dr.RasterizeGLContext()
render_resolution = [256, 256]


target_img = cv2.imread("images/pug.jpg").astype(np.float32) / 255.0
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
target_img = cv2.resize(target_img, render_resolution)
cv2.imwrite('check.png', cv2.cvtColor(target_img * 255, cv2.COLOR_RGB2BGR))
target_img = torch.Tensor(target_img).to(device)

epochs = 2000
# optimizer = optim.Adam([xy_coords], lr=0.001)
optimizer = optim.Adam([xy_coords, col], lr=0.001)
for i in range(epochs):
    optimizer.zero_grad()
    pos = torch.zeros([num_triangles * 3, 4], dtype=torch.float32).to(device)
    pos[:,0:2] = xy_coords
    pos[:, 3] = 1

    # col = torch.zeros([num_triangles * 3, 3], dtype=torch.float32).requires_grad_().to(device)
    # col[:, 0] = col_value[:,0]
    # col[:, 1] = col_value[:,0]
    # col[:, 2] = col_value[:,0]

    pos = torch.unsqueeze(pos, 0)
    rast, _ = dr.rasterize(glctx, pos, tri, resolution=render_resolution)
    out, _ = dr.interpolate(col, rast, tri)

    img = dr.antialias(out, rast, pos, tri)
    loss = torch.nn.functional.mse_loss(target_img.unsqueeze(0), img)
    print("epoch={}, loss={}".format(i, loss))
    loss.backward()
    optimizer.step()

save_img(img, "tri.png")

