import functions as f
import matplotlib.pyplot as plt

QUALITY = 25

SHOW = False

FACTOR = [4, 2, 0]

IMAGES = ["imagens\\barn_mountains.bmp",
          "imagens\\logo.bmp", "imagens\\peppers.bmp"]

FIG_SIZE = (12,7)


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
    if SHOW:
        plt.figure("Quantization", figsize=FIG_SIZE)
        f.visualize_Dct(Y_q, Cb_q, Cr_q)

    # DPCM
    Y_dpcm, Cb_dpcm, Cr_dpcm = f.dpcm(Y_q, Cb_q, Cr_q)
    if SHOW:
        plt.figure("DPCM", figsize=FIG_SIZE)
        f.visualize_Dct(Y_dpcm, Cb_dpcm, Cr_dpcm)

    return [Y_dpcm, Cb_dpcm, Cr_dpcm]


def decoder(YCbCr, original):

    # Inverse DPCM
    Y_idpcm, Cb_idpcm, Cr_idpcm = f.idpcm(YCbCr[0], YCbCr[1], YCbCr[2])
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
        f.visualize_RGB(R, G, B, "gray")

    image_padded = f.merge_RGB(R, G, B)

    # Faz unpadding
    unpadded = f.unpadding(image_padded, original)

    if SHOW:

        plt.figure("Imagem depois de unpadding", figsize=FIG_SIZE)

        plt.imshow(unpadded)

        plt.axis('off')

        plt.show()

    return unpadded


def main():

    for image_path in IMAGES:

        image = f.plt.imread(image_path)

        # Create a colormap (3.2)
        cmUser = f.colormap('cmUser', [(0, 0, 0), (1, 1, 1)], 256)

        # Visualize image with a colormap (3.3)
        # f.image_colormap(image, cmUser)

        blocks64 = encoder(
            image, cmUser)

        final = decoder(blocks64, image)

        plt.figure("Inicial vs Final", figsize=FIG_SIZE)
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Inicial")

        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(final, cmap="gray")
        plt.title("Final")

        plt.axis("off")

        plt.show()


if __name__ == '__main__':

    main()
