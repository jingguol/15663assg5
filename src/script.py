import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.ndimage
import math
import cv2
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

import cp_hw5

import sys
np.set_printoptions(threshold=sys.maxsize)

epsilon = 1e-5

## Initials
image = plt.imread('../data/shoe1.tiff')
height = image.shape[0]
width = image.shape[1]
height = 200
width = 300
P = height * width
I = np.zeros((7, P))
for i in range(1, 8) :
    image = plt.imread('../data/coke' + str(i) + '.tiff')
    # image = np.divide(image, 255.0, dtype=np.float32)
    image = skimage.transform.resize(image, (200, 300))
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    image = image[:, :, 1].flatten()
    I[i-1, :] = image

# np.savetxt('I_coke.npy', I)
# I = np.loadtxt('I_shoe.npy')

## Uncalibrated photometric stereo
U, S, V = np.linalg.svd(I, full_matrices=False)
L_e = U[:, 0:3].T
B_e = V[0:3, :]
for i in range(3) :
    L_e[i, :] *= np.sqrt(S[i])
    B_e[i, :] *= np.sqrt(S[i])
# Q = np.array([[0, 2, 0], [0, 0, -1], [0.5, 0, 0]], dtype=np.float32)
# B_e = np.matmul(np.linalg.inv(Q).T, B_e)
A_e = np.linalg.norm(B_e, ord=None, axis=0)
N_e = np.divide(B_e, np.vstack((A_e, A_e, A_e)))
albedo = np.reshape(A_e, (height, width))
normal = np.reshape(((N_e + 1.0) / 2.0).T, (height, width, 3))
# plt.imshow(albedo / np.max(albedo), cmap='gray')
# plt.show()
# plt.imsave('albedo_Q.png', albedo / np.max(albedo), cmap='gray')
# plt.imshow(np.clip(normal, 0.0, 1.0))
# plt.show()
# plt.imsave('normal_Q.png', np.clip(normal, 0.0, 1.0))


## Enforcing integrability
b_e = np.reshape(B_e.T, (height, width, 3))
b_e_blur = np.zeros(b_e.shape)
for i in range(3) :
    b_e_blur[:, :, i] = scipy.ndimage.gaussian_filter(b_e[:, :, i], 5)
b_e_x = np.gradient(b_e_blur, axis=0)
b_e_y = np.gradient(b_e_blur, axis=1)

b_e = np.reshape(b_e, (height * width, 3))
b_e_x = np.reshape(b_e_x, (height * width, 3))
b_e_y = np.reshape(b_e_y, (height * width, 3))
A = np.zeros((height * width, 6))
A[:, 0] = b_e[:, 0] * b_e_x[:, 1] - b_e[:, 1] * b_e_x[:, 0]
A[:, 1] = b_e[:, 0] * b_e_x[:, 2] - b_e[:, 2] * b_e_x[:, 0]
A[:, 2] = b_e[:, 1] * b_e_x[:, 2] - b_e[:, 2] * b_e_x[:, 1]
A[:, 3] = -b_e[:, 0] * b_e_y[:, 1] + b_e[:, 1] * b_e_y[:, 0]
A[:, 4] = -b_e[:, 0] * b_e_y[:, 2] + b_e[:, 2] * b_e_y[:, 0]
A[:, 5] = -b_e[:, 1] * b_e_y[:, 2] + b_e[:, 2] * b_e_y[:, 1]
U, S, V = np.linalg.svd(A, full_matrices=False)
x = V[-1, :]
delta = np.array([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]], dtype=np.float32)
G_F = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
B_e = np.matmul(delta, B_e)
B_e = np.matmul(G_F, B_e)
A_e = np.linalg.norm(B_e, ord=None, axis=0)
N_e = np.divide(B_e, np.vstack((A_e, A_e, A_e)))[[1, 0, 2], :]
albedo = np.reshape(A_e, (height, width))
normal = np.reshape(((N_e + 1.0) / 2.0).T, (height, width, 3))
# plt.imshow(albedo / np.max(albedo), cmap='gray')
# plt.show()
# plt.imsave('coke.png', albedo / np.max(albedo), cmap='gray')
# plt.imshow(np.clip(normal, 0.0, 1.0))
# plt.show()
# plt.imsave('coke.png', np.clip(normal, 0.0, 1.0))


# Normal integration
G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
B_e = np.matmul(G, B_e)
A_e = np.linalg.norm(B_e, ord=None, axis=0)
N_e = np.divide(B_e, np.vstack((A_e, A_e, A_e)))[[1, 0, 2], :]
albedo = np.reshape(A_e, (height, width))
normal = np.reshape(((N_e + 1.0) / 2.0).T, (height, width, 3))
# plt.imshow(albedo / np.max(albedo), cmap='gray')
# plt.show()
# plt.imsave('albedo_GBR.png', albedo / np.max(albedo), cmap='gray')
# plt.imshow(np.clip(normal, 0.0, 1.0))
# plt.show()
# plt.imsave('normal_GBR.png', np.clip(normal, 0.0, 1.0))

N_e[np.isnan(N_e)] = 0
N_e = np.reshape(N_e.T, (height, width, 3)) + epsilon
N_e = np.divide(N_e, np.stack((-N_e[:, :, 2], -N_e[:, :, 2], -N_e[:, :, 2]), axis=-1))
Z = cp_hw5.integrate_poisson(N_e[:, :, 0], N_e[:, :, 1])
# plt.imshow(1.0 - (Z / np.max(Z)), cmap='gray')
# plt.show()
# plt.imsave('depth.png', 1.0 - (Z / np.max(Z)), cmap='gray')

H, W = Z.shape
x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ls = LightSource()
color_shade = ls.shade(Z, plt.cm.gray)
surf = ax.plot_surface(x, y, Z, facecolors=color_shade, rstride=4, cstride=4)
plt.axis('off')
plt.show()

## Calibrated photometric stereo
L = cp_hw5.load_sources()
B = np.linalg.inv(L.T @ L) @ L.T @ I
G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
B_e = np.matmul(G, B)
A_e = np.linalg.norm(B_e, ord=None, axis=0)
N_e = np.divide(B_e, np.vstack((A_e, A_e, A_e)))[[1, 0, 2], :]
albedo = np.reshape(A_e, (height, width))
normal = np.reshape(((N_e + 1.0) / 2.0).T, (height, width, 3))
# plt.imshow(albedo / np.max(albedo), cmap='gray')
# plt.show()
# plt.imsave('albedo_calibrated.png', albedo / np.max(albedo), cmap='gray')
# plt.imshow(np.clip(normal, 0.0, 1.0))
# plt.show()
# plt.imsave('normal_calibrated.png', np.clip(normal, 0.0, 1.0))

# H, W = Z.shape
# x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ls = LightSource()
# color_shade = ls.shade(Z, plt.cm.gray)
# surf = ax.plot_surface(x, y, Z, facecolors=color_shade, rstride=4, cstride=4)
# plt.axis('off')
# plt.show()

