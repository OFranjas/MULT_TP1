import functions as f
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

QUALITY = 10

SHOW = False

FACTOR = [4, 2, 0]

IMAGES = ["imagens\\barn_mountains.bmp",
          "imagens\\logo.bmp", "imagens\\peppers.bmp"]

FIG_SIZE = (12, 7)


def encoder(image, colormap):

    # Padding of the image (4.1)
    image_padded = f.padding(image)

    if SHOW:

        plt.figure("Imagem com padding", figsize=FIG_SIZE)

        plt.imshow(image_padded, cmap=colormap)

        plt.title("Imagem com padding")

        plt.axis('off')

        plt.show()

    # Separate the image in 3 channels (3.4)
    R, G, B = f.separate_RGB(image_padded)

    if SHOW:
        # Visualize the image and each one of its channels (with the adequate colormap) (3.5)
        f.visualize_RGB(R, G, B, "gray", FIG_SIZE)

    # Convert image from RGB to YCbCr (5.1)
    Y, Cb, Cr = f.RGB_to_YCbCr(R, G, B)

    # Downsample the channels
    Y_d, Cb_d, Cr_d = f.downsamplingV2(Y, Cb, Cr, FACTOR)

    # Show image with downsampled channels
    if SHOW:
        plt.figure("Downsampled channels", figsize=FIG_SIZE)
        # downsampled = f.merge_RGB(Y_d, Cb_d, Cr_d)
        f.visualize_YCbCr(Y_d, Cb_d, Cr_d, "gray")
        plt.show()

    # Blocks DCT
    block = 8
    Y_b = f.blocks_Dct(Y_d, size=block)
    Cb_b = f.blocks_Dct(Cb_d, size=block)
    Cr_b = f.blocks_Dct(Cr_d, size=block)
    if SHOW:
        plt.figure("Block DCT (8x8)", figsize=FIG_SIZE)
        f.visualize_Dct(Y_b, Cb_b, Cr_b)

    # Quantization
    Y_q, Cb_q, Cr_q = f.quantization(Y_b, Cb_b, Cr_b, QUALITY)
    print("Quantização - ", Y_q[8:16, 8:16])
    if SHOW:
        plt.figure("Quantization", figsize=FIG_SIZE)
        f.visualize_Dct(Y_q, Cb_q, Cr_q)

    # DPCM
    Y_dpcm, Cb_dpcm, Cr_dpcm = f.dpcm(Y_q, Cb_q, Cr_q)
    print("DPCM - ", Y_dpcm[8:16, 8:16])

    if SHOW:
        plt.figure("DPCM", figsize=FIG_SIZE)
        f.visualize_Dct(Y_dpcm, Cb_dpcm, Cr_dpcm)

    return Y_dpcm, Cb_dpcm, Cr_dpcm, image.shape, Y


def decoder(YCbCr, original):

    # Inverse DPCM
    Y_idpcm, Cb_idpcm, Cr_idpcm = f.idpcm(YCbCr[0], YCbCr[1], YCbCr[2])
    print("IDPCM - ", Y_idpcm[8:16, 8:16])

    if SHOW:
        plt.figure("Inverse DPCM", figsize=FIG_SIZE)
        f.visualize_Dct(Y_idpcm, Cb_idpcm, Cr_idpcm)

    # Dequantization
    Y_dq, Cb_dq, Cr_dq = f.desquantization(
        Y_idpcm, Cb_idpcm, Cr_idpcm, QUALITY)
    if SHOW:
        plt.figure("Dequantization", figsize=FIG_SIZE)
        f.visualize_Dct(Y_dq, Cb_dq, Cr_dq)

    # Inverse Block DCT
    block = 8
    Y_b = f.blocks_Idct(Y_dq, size=block)
    Cb_b = f.blocks_Idct(Cb_dq, size=block)
    Cr_b = f.blocks_Idct(Cr_dq, size=block)
    if SHOW:
        plt.figure("Block Inverse DCT", figsize=FIG_SIZE)
        # image = f.merge_RGB(Y_b, Cb_b, Cr_b)
        f.visualize_YCbCr(Y_b, Cb_b, Cr_b, "gray")
        plt.show()

    # Up-sample the channels
    Y_u, Cb_u, Cr_u = f.upsamplingV2(
        Y_b, Cb_b, Cr_b, FACTOR)

    # Show image with upsampled channels
    if SHOW:
        plt.figure("Upsampled channels", figsize=FIG_SIZE)
        # upsampled = f.merge_RGB(Y_u, Cb_u, Cr_u)
        f.visualize_YCbCr(Y_u, Cb_u, Cr_u, "gray")
        plt.show()

    # Transforma de volta RGB
    R, G, B = f.YCbCr_to_RGB(
        Y_u, Cb_u, Cr_u)

    # Visualiza imagem com canais separados
    if SHOW:
        f.visualize_RGB(R, G, B, "gray", FIG_SIZE)

    image_padded = f.merge_RGB(R, G, B)

    # Faz unpadding
    unpadded = f.unpadding(image_padded, original)

    if SHOW:

        plt.figure("Imagem depois de unpadding", figsize=FIG_SIZE)

        plt.imshow(unpadded)

        plt.axis('off')

        plt.show()

    return unpadded, Y_u


def metricas(filepath: str, qf: int = 75, show: bool = True, met: bool = True) -> np.ndarray:
    original = np.array(Image.open(filepath))
    y, cb, cr, shape, yOriginal = encoder(original, "gray")
    compressed, yReconstructed = decoder((y, cb, cr), original)
    diff = np.absolute(yOriginal - yReconstructed)
    print("Diferença max - ", np.max(diff))
    if show:
        plt.figure("Comprimida vs Diferença", FIG_SIZE)
        plt.subplot(1, 2, 1)
        plt.title("Comprimida")
        f.visualize_image(compressed)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Diferença")
        plt.axis('off')
        f.visualize_image(diff, cmap="gray")
        plt.show()
    mse = f.MSE(original, compressed)
    rmse = f.RMSE(mse)
    snr = f.SNR(original, mse)
    psnr = f.PSNR(original, mse)
    if met:
        print(f"QF: {qf}")
        print(f"MSE: {mse:.3f}\nRMSE: {rmse:.3f}")
        print(f"SNR: {snr:.3f} dB\nPSNR: {psnr:.3f} dB")
    plt.show()
    return compressed, {'mse': mse, 'rmse': rmse, 'snr': snr, 'psnr': psnr}


def main():

    count = 0

    for image_path in IMAGES:

        count += 1

        image = f.plt.imread(image_path)

        # Create a colormap (3.2)
        cmUser = f.colormap('cmUser', [(0, 0, 0), (1, 1, 1)], 256)

        # Visualize image with a colormap (3.3)
        # f.image_colormap(image, cmUser)

        blocks64 = encoder(
            image, cmUser)

        final, Y_u = decoder(blocks64, image)

        # Show Initial vs Final
        plt.figure("Initial vs Final", figsize=FIG_SIZE)

        plt.subplot(1, 2, 1)
        plt.title("Initial")
        plt.imshow(image, cmap="gray")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Final")
        plt.imshow(final, cmap="gray")
        plt.axis('off')
        plt.show()

        # Codec
        print(f"Imagem {count}:")
        metricas(image_path, qf=QUALITY, show=True, met=True)


if __name__ == '__main__':

    main()
