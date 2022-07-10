# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pickletools import optimize
from xml.etree.ElementTree import tostringlist
import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr
from torch import optim
import cv2
import torch.nn.functional as F
from numpy import random

import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

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

def normalize_image(img):
    # img: nchw
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).reshape(1, 3, 1, 1)
    mean.expand(1, 3, img.shape[2], img.shape[3])
    std.expand(1, 3, img.shape[2], img.shape[3])
    return (img - mean) / std

def custom_transform(image, size):
    image = torch.nn.functional.interpolate(image, (size, size), mode='bilinear', align_corners=False)
    image = normalize_image(image)
    return image

num_triangles = 200
boundary = 0.8

xy_coords = torch.zeros([num_triangles * 3, 2], dtype=torch.float32).to(device)

for i in range(num_triangles):
    center = (random.uniform(-boundary, boundary), random.uniform(-boundary, boundary))
    radius = random.uniform(0.2, 0.4)
    theta = random.uniform(0, np.pi)
    for j in range(3):
        coord = get_coord(theta + j * 2.0 * np.pi / 3.0, center, radius)
        xy_coords[3 * i + j, 0], xy_coords[3 * i + j, 1] = coord[0], coord[1]
        arr_col.append([0.5, 0.5, 0.5])
    arr_tri.append([3 * i, 3 * i + 1, 3 * i + 2])

col = tensor([arr_col], dtype=torch.float32).requires_grad_().to(device)
tri = tensor(arr_tri, dtype=torch.int32)
xy_coords.requires_grad_()

# CLIP
model, preprocess = clip.load('ViT-B/32', device)
clip_input_size = model.visual.input_resolution
text = clip.tokenize(["dog"]).to(device)

glctx = dr.RasterizeGLContext()
render_resolution = [clip_input_size, clip_input_size]

# target_img = cv2.imread("images/canada.png").astype(np.float32) / 255.0
# target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
# target_img = cv2.resize(target_img, render_resolution)
# cv2.imwrite('check.png', cv2.cvtColor(target_img * 255, cv2.COLOR_RGB2BGR))
# target_img = torch.Tensor(target_img).to(device)
class Net(torch.nn.Module):
    def __init__(self, ntrigs, ncols) -> None:
        super().__init__()
        self.features = torch.nn.Parameter(torch.zeros((1,)))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,ntrigs+ncols),
        )
        self.trigs = ntrigs
        self.cols = ncols

    def forward(self):
        y = torch.tanh(self.net(self.features))
        trigs = y[:self.trigs]
        cols = y[self.trigs:]
        return trigs, cols * 0.5 + 0.5

def gaussian_kernel(w, sigma=1):
    assert w % 2 == 1
    h = w // 2
    g = torch.zeros((w,w)).float()
    for y in range(-h, h+1):
        for x in range(-h, h+1):
            fx = x / h
            fy = y / h
            g[x+h][y+h] = np.exp(-(fx*fx+fy*fy)/(2*sigma*sigma))
    g /= torch.sum(g)
    g =  g.to(device)
    g = g.reshape(1,1,w,w)
    return g.repeat(3,3,1,1)

target_img = cv2.imread("images/canada.png").astype(np.float32) / 255.0
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
target_img = cv2.resize(target_img, render_resolution)
cv2.imwrite('check.png', cv2.cvtColor(target_img * 255, cv2.COLOR_RGB2BGR))
target_img = torch.Tensor(target_img).to(device)

epochs = 1000
net = Net(num_triangles*3*2, num_triangles * 3 * 3).to(device)
# optimizer = optim.Adam([xy_coords], lr=0.001)
# optimizer = optim.Adam([xy_coords, col], lr=0.001)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# blur_kernel = gaussian_kernel(5, 1)
for i in range(epochs):
    optimizer.zero_grad()
    pos = torch.zeros([num_triangles * 3, 4], dtype=torch.float32).to(device)
    xy_coords_offset, col = net()
    col = col.reshape(1,num_triangles*3,3)
    pos[:,0:2] = xy_coords_offset.reshape(num_triangles*3,2)# * 0.3 + xy_coords
    pos[:, 3] = 1

    pos = torch.unsqueeze(pos, 0)
    rast, _ = dr.rasterize(glctx, pos, tri, resolution=render_resolution)
    out, _ = dr.interpolate(col, rast, tri)
    img = dr.antialias(out, rast, pos, tri)
    # img = img.nan_to_num()
    # img = torch.clamp(img, 0.0, 1.0)
    # img = out
    # clip
    # proc_image = preprocess(img).unsqueeze(0).to(device)
    proc_image = img.permute((0, 3, 1, 2))
    # proc_image = proc_image[:,[2,1,0],:,:]
    # proc_image = F.conv2d(proc_image, blur_kernel, padding=2)

    # proc_image = F.interpolate(proc_image, size=(16,16),mode='area')
    # proc_image = F.interpolate(proc_image, size=(clip_input_size,clip_input_size),mode='bilinear')
    with torch.no_grad():
        cv2.imshow('img', proc_image.detach().squeeze().permute(1,2,0).cpu().numpy())
        cv2.waitKey(1)
    # with torch.no_grad():
    proc_image = normalize_image(proc_image).to(device)
    image_features = model.encode_image(proc_image)
    text_features = model.encode_text(text)
    loss = 1.0 - torch.nn.functional.cosine_similarity(image_features, text_features)
    # print('loss=', clip_loss)
    # loss = torch.nn.functional.mse_loss(target_img.unsqueeze(0), img)
    print("epoch={}, loss={}".format(i, loss.item()))
    loss.backward()
    # print(col,'\n', col.grad)
    optimizer.step()
    if i % 100 == 0:
        save_img(img, "tri_{}.png".format(i))

save_img(img, "tri.png")

