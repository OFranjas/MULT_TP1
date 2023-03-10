# Create a colormap (3.2)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import fftpack as fft
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


def visualize_image(image: np.ndarray, **kwargs: dict[str, any]) -> None:
    # Get keyword arguments
    cmap = kwargs.get("cmap", None)

    # Create a new figure to display the image
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
    plt.imshow(image, cmap=cmap)


# Visualize the image and each one of its channels (with the adequate colormap) (3.5)
# Gray colormap for the channels
def visualize_RGB(R, G, B, colormapx, size):

    plt.figure("RGB channels", figsize=size)

    cmRed = colormap("Red", [(0, 0, 0), (1, 0, 0)], 256)

    cmBlue = colormap("Blue", [(0, 0, 0), (0, 0, 1)], 256)

    cmGreen = colormap("Green", [(0, 0, 0), (0, 1, 0)], 256)

    plt.subplot(2, 2, 1)
    plt.imshow(R, cmap=cmRed)
    plt.axis('off')
    plt.title('R')

    plt.subplot(2, 2, 2)
    plt.imshow(G, cmap=cmGreen)
    plt.axis('off')
    plt.title('G')

    plt.subplot(2, 2, 3)
    plt.imshow(B, cmap=cmBlue)
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

    return image_nova


def unpadding(image, origin):

    [nl, nc, x] = origin.shape

    original_image = image[:nl, :nc]

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


def visualize_YCbCr(Y, Cb, Cr, colormap):

    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap=colormap)
    plt.axis('off')
    plt.title('Y')

    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap=colormap)
    plt.axis('off')
    plt.title('Cb')

    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap=colormap)
    plt.axis('off')
    plt.title('Cr')

    plt.show()


def downsamplingV2(Y, Cb, Cr, factor):

    if factor == [4, 4, 4]:

        return Y, Cb, Cr

    resCb, resCr = Cb, Cr

    horizontal = False

    perc_Cb = factor[1]/factor[0]

    if factor[2] == 0:

        perc_Cr = perc_Cb

        horizontal = True

    else:

        perc_Cr = factor[2]/factor[0]

    if horizontal:

        resCb = cv2.resize(Cb, None, perc_Cb, perc_Cb, cv2.INTER_AREA)

        resCr = cv2.resize(Cr, None, perc_Cr, perc_Cr, cv2.INTER_AREA)

    else:

        resCb = cv2.resize(Cb, None, 1, perc_Cb, cv2.INTER_AREA)

        resCr = cv2.resize(Cr, None, 1, perc_Cr, cv2.INTER_AREA)

    return Y, resCb, resCr


def upsamplingV2(Y, Cb, Cr, factor):

    size = Y.shape[::-1]

    resCb = cv2.resize(Cb, size, cv2.INTER_CUBIC)
    resCr = cv2.resize(Cr, size, cv2.INTER_CUBIC)

    return Y, resCb, resCr


def dct(X):
    return fft.dct(fft.dct(X, norm="ortho").T, norm="ortho").T


def idct(X):
    return fft.idct(fft.idct(X, norm="ortho").T, norm="ortho").T


def visualize_Dct(x1, x2, x3):
    x1log = np.log(np.abs(x1) + 0.0001)
    x2log = np.log(np.abs(x2) + 0.0001)
    x3log = np.log(np.abs(x3) + 0.0001)

    # image = merge_RGB(x1log, x2log, x3log)

    visualize_YCbCr(x1log, x2log, x3log, "gray")

    # plt.subplot(2, 2, 4)
    # plt.imshow(merge_RGB(x1log, x2log, x3log))
    # plt.title("DCT")
    # plt.axis("off")
    # plt.show()


def blocks_Dct(x, size=8):
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = dct(x[i:i+size, j:j+size])
    return newImg


def blocks_Idct(x, size=8):
    h, w = x.shape
    newImg = np.empty(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
    return newImg


def quantization(Y, Cb, Cr, quality):

    QY = np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 35, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]])

    if (quality >= 50):

        x = (100-quality)/50

    else:
        x = 50/quality

    QY = np.round(QY*x)
    QC = np.round(QC*x)

    QY[QY > 255] = 255
    QC[QC > 255] = 255

    QY[QY < 1] = 1
    QC[QC < 1] = 1

    QC = QC.astype(np.uint8)
    QY = QY.astype(np.uint8)

    resY = np.empty(Y.shape, dtype=Y.dtype)
    resCb = np.empty(Cb.shape, dtype=Cb.dtype)
    resCr = np.empty(Cr.shape, dtype=Cr.dtype)

    for i in range(0, Y.shape[0], 8):

        for j in range(0, Y.shape[1], 8):

            resY[i:i+8, j:j+8] = Y[i:i+8, j:j+8]/QY

    resY = np.round(resY).astype(int)

    for i in range(0, Cb.shape[0], 8):

        for j in range(0, Cb.shape[1], 8):

            resCb[i:i+8, j:j+8] = Cb[i:i+8, j:j+8]/QC

    resCb = np.round(resCb).astype(int)

    for i in range(0, Cr.shape[0], 8):

        for j in range(0, Cr.shape[1], 8):

            resCr[i:i+8, j:j+8] = Cr[i:i+8, j:j+8]/QC

    resCr = np.round(resCr).astype(int)

    return resY, resCb, resCr


def desquantization(Y, Cb, Cr, quality):

    QY = np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 35, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]])

    if (quality >= 50):

        x = (100-quality)/50
    else:
        x = 50/quality

    QY = np.round(QY*x)
    QC = np.round(QC*x)

    QY[QY > 255] = 255
    QC[QC > 255] = 255

    QY[QY < 1] = 1
    QC[QC < 1] = 1

    QY = QY.astype(np.uint8)
    QC = QC.astype(np.uint8)

    resY = np.empty(Y.shape)
    resCb = np.empty(Cb.shape)
    resCr = np.empty(Cr.shape)

    for i in range(0, Y.shape[0], 8):

        for j in range(0, Y.shape[1], 8):

            resY[i:i+8, j:j+8] = Y[i:i+8, j:j+8]*QY

    resY = np.round(resY).astype(float)

    for i in range(0, Cb.shape[0], 8):

        for j in range(0, Cb.shape[1], 8):

            resCb[i:i+8, j:j+8] = Cb[i:i+8, j:j+8]*QC

    resCb = np.round(resCb).astype(float)

    for i in range(0, Cr.shape[0], 8):

        for j in range(0, Cr.shape[1], 8):

            resCr[i:i+8, j:j+8] = Cr[i:i+8, j:j+8]*QC

    resCr = np.round(resCr).astype(float)

    return resY, resCb, resCr


def dpcm(Y, Cb, Cr):

    # Y
    dc = Y[0, 0]

    l, c = Y.shape

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Y[i, j]
            dif = aux - dc
            dc = aux
            Y[i, j] = dif

    # Cb
    dc = Cb[0, 0]

    l, c = Cb.shape

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Cb[i, j]
            dif = aux - dc
            dc = aux
            Cb[i, j] = dif

    # CR
    dc = Cr[0, 0]

    l, c = Cr.shape

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Cr[i, j]
            dif = aux - dc
            dc = aux
            Cr[i, j] = dif

    return Y, Cb, Cr


def idpcm(Y, Cb, Cr):

    # Y

    l, c = Y.shape

    dc = Y[0, 0]

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Y[i, j]
            soma = dc + aux

            dc = soma

            Y[i, j] = soma

    # Cb

    l, c = Cb.shape

    dc = Cb[0, 0]

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Cb[i, j]
            soma = dc + aux

            dc = soma

            Cb[i, j] = soma

    # Cr

    l, c = Cr.shape

    dc = Cr[0, 0]

    for i in range(0, l, 8):

        for j in range(0, c, 8):

            if i == 0 and j == 0:
                continue

            aux = Cr[i, j]
            soma = dc + aux

            dc = soma

            Cr[i, j] = soma

    return Y, Cb, Cr


def MSE(x: np.ndarray, y: np.ndarray) -> np.float64:
    h, w = x[:, :, 0].shape
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    coef = 1 / (h * w)
    return coef * (np.sum((x - y) ** 2))


def RMSE(mse: np.float64) -> np.float64:
    return mse ** (1 / 2)


def SNR(x: np.ndarray, mse: np.float64) -> np.float64:
    h, w = x[:, :, 0].shape
    x = x.astype(np.float64)
    coef = 1 / (w * h)
    power = coef * np.sum(x ** 2)
    return 10 * np.log10(power / mse)


def PSNR(x: np.ndarray, mse: np.float64) -> np.float64:
    return 10 * np.log10((np.max(x) ** 2) / mse)
