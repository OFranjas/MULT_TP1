# Create a colormap (3.2)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import fftpack as fft
import cv2
import scipy.fftpack as scp


def colormap(name, colors, N):

    cm = clr.LinearSegmentedColormap.from_list(name, colors, N)

    return cm

# Visualize image with a colormap (3.3)


def image_colormap(image, cm):

    plt.figure()

    plt.imshow(image, cmap=cm)

    plt.axis('off')

    plt.title('Imagem com colormap')

    plt.colorbar()

    plt.show()


# Separate the image in 3 channels (3.4)

def separate_RGB(image):

    R = image[:, :, 0]

    G = image[:, :, 1]

    B = image[:, :, 2]

    return R, G, B


# Merge the 3 channels into an image (3.4)

def merge_RGB(R, G, B):

    image = np.dstack((R, G, B))

    return image


# Visualize the image and each one of its channels (with the adequate colormap) (3.5)
# Gray colormap for the channels
def visualize_RGB(R, G, B):

    cmGray = colormap('cmGray', [(0, 0, 0), (1, 1, 1)], 256)

    plt.figure("RGB channels")

    plt.subplot(2, 2, 1)
    plt.imshow(R, cmap=cmGray)
    plt.axis('off')
    plt.title('R')

    plt.subplot(2, 2, 2)
    plt.imshow(G, cmap=cmGray)
    plt.axis('off')
    plt.title('G')

    plt.subplot(2, 2, 3)
    plt.imshow(B, cmap=cmGray)
    plt.axis('off')
    plt.title('B')

    plt.subplot(2, 2, 4)
    plt.imshow(merge_RGB(R, G, B))
    plt.axis('off')
    plt.title('Merged Channels')
    plt.show()


# Padding of the image (4.1)


def padding(image):

    [nl, nc, x] = image.shape

    nnl = 32 - nl % 32/589

    nnc = 32 - nc % 32

    ll = image[nl-1, :][np.newaxis, :]  # Horizontalmente

    rep = ll.repeat(nnl, axis=0)

    image_nova = np.vstack([image, rep])

    cc = image_nova[:, nc-1][:, np.newaxis]  # Verticalmente

    rep = cc.repeat(nnc, axis=1)

    image_nova = np.hstack([image_nova, rep])

    plt.imshow(image_nova)

    plt.title("Imagem com padding")

    plt.axis('off')

    plt.show()

    return image_nova


def unpadding(image, origin):

    [nl, nc, x] = origin.shape

    original_image = image[:nl, :nc]

    plt.imshow(original_image)

    plt.title("Imagem depois de unpadding")

    plt.axis('off')

    plt.show()

    return original_image


# Convert image from RGB to YCbCr (5.1)


def RGB_to_YCbCr(R, G, B):

    T = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])

    Y = T[0, 0]*R + T[0, 1]*G + T[0, 2]*B
    Cb = T[1, 0]*R + T[1, 1]*G + T[1, 2]*B + 128
    Cr = T[2, 0]*R + T[2, 1]*G + T[2, 2]*B + 128

    return Y, Cb, Cr

# Convert image from YCbCr to RGB (5.1)


def YCbCr_to_RGB(Y, Cb, Cr):

    T = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])

    T_inv = np.linalg.inv(T)

    R = T_inv[0, 0]*(Y - 0) + T_inv[0, 1]*(Cb - 128) + T_inv[0, 2]*(Cr - 128)

    R[R > 255] = 255
    R[R < 0] = 0
    R = np.round(R).astype(np.uint8)

    G = T_inv[1, 0]*(Y - 0) + T_inv[1, 1]*(Cb - 128) + T_inv[1, 2]*(Cr - 128)

    G[G > 255] = 255
    G[G < 0] = 0
    G = np.round(G).astype(np.uint8)

    B = T_inv[2, 0]*(Y - 0) + T_inv[2, 1]*(Cb - 128) + T_inv[2, 2]*(Cr - 128)

    B[B > 255] = 255
    B[B < 0] = 0
    B = np.round(B).astype(np.uint8)

    return R, G, B

# Visualize the image and its YCbCr components (5.2)


def visualize_YCbCr(Y, Cb, Cr):

    cmGray = colormap('cmGray', [(0, 0, 0), (1, 1, 1)], 256)

    plt.subplot(2, 2, 1)
    plt.imshow(Y, cmap=cmGray)
    plt.axis('off')
    plt.title('Y')

    plt.subplot(2, 2, 2)
    plt.imshow(Cb, cmap=cmGray)
    plt.axis('off')
    plt.title('Cb')

    plt.subplot(2, 2, 3)
    plt.imshow(Cr, cmap=cmGray)
    plt.axis('off')
    plt.title('Cr')

    # plt.show()


# Downsampling of the image (6.1)
def downsampling(Y, Cb, Cr, factor):

    if factor[2] == 0:

        #percent_y = factor[0]/factor[1]
        percent_cb = factor[0]/factor[1]
        percent_cr = factor[0]/factor[1]

        #y_d = int(Y.shape[1] / percent_y)
        cb_d = int(Cb.shape[1] / percent_cb)
        cr_d = int(Cr.shape[1] / percent_cr)
        #y_dy = int(Y.shape[0] / percent_y)
        cb_dy = int(Cb.shape[0] / percent_cb)
        cr_dy = int(Cr.shape[0] / percent_cr)

        # Y_down = cv2.resize(
        #     Y, (y_d, y_dy), interpolation=cv2.INTER_AREA)

        Cb_down = cv2.resize(
            Cb, (cb_d, cb_dy), interpolation=cv2.INTER_AREA)

        Cr_down = cv2.resize(
            Cr, (cr_d, cr_dy), interpolation=cv2.INTER_AREA)

        return Y, Cb_down, Cr_down

    elif factor[1] == 0:

        print("Error: Factor 1 is 0")

    else:

        percent_y = factor[0]/factor[1]
        percent_cb = factor[0]/factor[1]
        percent_cr = factor[0]/factor[2]

        y_d = int(Y.shape[1]/percent_y)
        cb_d = int(Cb.shape[1]/percent_cb)
        cr_d = int(Cr.shape[1]/percent_cr)

        Y_down = cv2.resize(
            src=Y, dsize=(y_d, Y.shape[0]), interpolation=cv2.INTER_AREA)

        Cb_down = cv2.resize(
            src=Cb, dsize=(cb_d, Y.shape[0]), interpolation=cv2.INTER_AREA)

        Cr_down = cv2.resize(
            src=Cr, dsize=(cr_d, Y.shape[0]), interpolation=cv2.INTER_AREA)

        return Y, Cb_down, Cr_down


# Upsampling of the image (6.1)
def upsampling(Y, Cb_d, Cr_d, factor, shape):

    if factor[2] == 0:

        #percent_y = factor[0]/factor[1]
        percent_cb = factor[0]/factor[1]
        percent_cr = factor[0]/factor[1]

        #y_u = int(Y_d.shape[1] * percent_y)
        cb_u = int(Cb_d.shape[1] * percent_cb)
        cr_u = int(Cr_d.shape[1] * percent_cr)
        #y_uy = int(Y_d.shape[0] * percent_y)
        cb_uy = int(Cb_d.shape[0] * percent_cb)
        cr_uy = int(Cr_d.shape[0] * percent_cr)

        if (shape[1] % 2 != 0):
            #y_u += 1
            cb_u += 1
            cr_u += 1

        if (shape[0] % 2 != 0):
            #y_uy += 1
            cb_uy += 1
            cr_uy += 1

        # Y_up = cv2.resize(
        #     Y_d, (y_u, y_uy), interpolation=cv2.INTER_CUBIC)

        Cb_up = cv2.resize(
            Cb_d, (cb_u, cb_uy), interpolation=cv2.INTER_CUBIC)

        Cr_up = cv2.resize(
            Cr_d, (cr_u, cr_uy), interpolation=cv2.INTER_CUBIC)

        return Y, Cb_up, Cr_up

    elif factor[1] == 0:

        print("Error: Factor 1 is 0")

    else:

        #percent_y = factor[0]/factor[1]
        percent_cb = factor[0]/factor[1]
        percent_cr = factor[0]/factor[2]

        #y_u = int(Y_d.shape[1] * percent_y)
        cb_u = int(Cb_d.shape[1] * percent_cb)
        cr_u = int(Cr_d.shape[1] * percent_cr)

        if (shape[1] % 2 != 0):
            #y_u += 1
            cb_u += 1
            cr_u += 1

        # Y_up = cv2.resize(
        #     src=Y_d, dsize=(y_u, Y_d.shape[0]), interpolation=cv2.INTER_CUBIC)

        Cb_up = cv2.resize(
            src=Cb_d, dsize=(cb_u, Y.shape[0]), interpolation=cv2.INTER_CUBIC)

        Cr_up = cv2.resize(
            src=Cr_d, dsize=(cr_u, Y.shape[0]), interpolation=cv2.INTER_CUBIC)

        return Y, Cb_up, Cr_up


def dct(X: np.ndarray) -> np.ndarray:
    return fft.dct(fft.dct(X, norm="ortho").T, norm="ortho").T


def idct(X: np.ndarray) -> np.ndarray:
    return fft.idct(fft.idct(X, norm="ortho").T, norm="ortho").T


def visualize_Dct(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> None:
    x1log = np.log(np.abs(x1) + 0.0001)
    x2log = np.log(np.abs(x2) + 0.0001)
    x3log = np.log(np.abs(x3) + 0.0001)

    #image = merge_RGB(x1log, x2log, x3log)

    visualize_YCbCr(x1log, x2log, x3log)

    plt.subplot(2, 2, 4)
    plt.imshow(merge_RGB(x1log, x2log, x3log))
    plt.title("DCT")
    plt.axis("off")
    plt.show()


def blocks_Dct(x: np.ndarray, size: int = 8) -> np.ndarray:
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = dct(x[i:i+size, j:j+size])
    return newImg


def blocks_Idct(x: np.ndarray, size: int = 8) -> np.ndarray:
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
    return newImg
