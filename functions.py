# Create a colormap (3.2)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2


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

    plt.figure()

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
    plt.title('Original')
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


def visualize_YCbCr(Y, Cb, Cr, image):

    cmGray = colormap('cmGray', [(0, 0, 0), (1, 1, 1)], 256)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(Y, cmap=cmGray)
    plt.axis('off')
    plt.title('Y')

    plt.subplot(2, 2, 2)
    plt.imshow(Cb, cmap="Blues")
    plt.axis('off')
    plt.title('Cb')

    plt.subplot(2, 2, 3)
    plt.imshow(Cr, cmap="Blues")
    plt.axis('off')
    plt.title('Cr')

    plt.subplot(2, 2, 4)
    plt.imshow(image, cmap="Blues")
    plt.axis('off')
    plt.title('Original')

    plt.show()


# Sub Sampling of the image's YCbCr components (6.1) Downsampling using cv2.resize

# def downsampling(Y, Cb, Cr):

#     Y_d = cv2.resize(Y, (0, 0), fx=0.5, fy=0.5,
#                      interpolation=cv2.INTER_NEAREST)

#     Cb_d = cv2.resize(Cb, (0, 0), fx=0.5, fy=0.5,
#                       interpolation=cv2.INTER_NEAREST)

#     Cr_d = cv2.resize(Cr, (0, 0), fx=0.5, fy=0.5,
#                       interpolation=cv2.INTER_NEAREST)

#     return Y_d, Cb_d, Cr_d


# def upsampling(Y_d, Cb_d, Cr_d):

#     Y_u = cv2.resize(Y_d, (0, 0), fx=2, fy=2,
#                      interpolation=cv2.INTER_NEAREST)

#     Cb_u = cv2.resize(Cb_d, (0, 0), fx=2, fy=2,
#                       interpolation=cv2.INTER_NEAREST)

#     Cr_u = cv2.resize(Cr_d, (0, 0), fx=2, fy=2,
#                       interpolation=cv2.INTER_NEAREST)

#     return Y_u, Cb_u, Cr_u
