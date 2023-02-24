import functions as f
import matplotlib.pyplot as plt


def encoder(image, factor, colormap, show):

    # Padding of the image (4.1)
    image_padded = f.padding(image)

    # Separate the image in 3 channels (3.4)
    R, G, B = f.separate_RGB(image_padded)

    # Visualize the image and each one of its channels (with the adequate colormap) (3.5)
    f.visualize_RGB(R, G, B, "gray")

    # Convert image from RGB to YCbCr (5.1)
    Y, Cb, Cr = f.RGB_to_YCbCr(R, G, B)

    # Downsample the channels
    Y_d, Cb_d, Cr_d = f.downsampling(Y, Cb, Cr, factor)

    # Show image with downsampled channels
    if show:
        plt.figure("Downsampled channels")
        # downsampled = f.merge_RGB(Y_d, Cb_d, Cr_d)
        f.visualize_YCbCr(Y_d, Cb_d, Cr_d, "winter")
        plt.show()

    # Whole-image DCT
    Y_w = f.dct(Y_d)
    Cb_w = f.dct(Cb_d)
    Cr_w = f.dct(Cr_d)
    if show:
        plt.figure("Whole-image DCT")
        f.visualize_Dct(Y_w, Cb_w, Cr_w)

    # Blocks DCT
    block = 8
    Y_b = f.blocks_Dct(Y_w, size=block)
    Cb_b = f.blocks_Dct(Cb_w, size=block)
    Cr_b = f.blocks_Dct(Cr_w, size=block)
    if show:
        plt.figure("Block DCT (8x8)")
        f.visualize_Dct(Y_b, Cb_b, Cr_b)

    # 64 x 64 blocks DCT
    block = 64
    dY = f.blocks_Dct(Y_b, size=block)
    dCb = f.blocks_Dct(Cb_b, size=block)
    dCr = f.blocks_Dct(Cr_b, size=block)
    if show:
        plt.figure("Block DCT (64x64)")
        f.visualize_Dct(dY, dCb, dCr)

    return [dY, dCb, dCr]


def decoder(blocks64, factor, original, show):

    # Inverse 64 Block DCT
    block = 64
    dY = f.blocks_Idct(blocks64[0], size=block)
    dCb = f.blocks_Idct(blocks64[1], size=block)
    dCr = f.blocks_Idct(blocks64[2], size=block)
    if show:
        plt.figure("Block Inverse DCT (64x64)")
        # image = f.merge_RGB(dY, dCb, dCr)
        f.visualize_YCbCr(dY, dCb, dCr, "winter")
        plt.show()

    # Inverse Block DCT
    block = 8
    Y_b = f.blocks_Idct(dY, size=block)
    Cb_b = f.blocks_Idct(dCb, size=block)
    Cr_b = f.blocks_Idct(dCr, size=block)
    if show:
        plt.figure("Block Inverse DCT")
        # image = f.merge_RGB(Y_b, Cb_b, Cr_b)
        f.visualize_YCbCr(Y_b, Cb_b, Cr_b, "winter")
        plt.show()

    # Whole-image inverse DCT

    Y_w = f.idct(Y_b)
    Cb_w = f.idct(Cb_b)
    Cr_w = f.idct(Cr_b)
    if show:
        plt.figure("Whole-image Inverse DCT")
        # inverse_w = f.merge_RGB(Y_w, Cb_w, Cr_w)
        f.visualize_YCbCr(Y_w, Cb_w, Cr_w, "winter")
        plt.show()

    # Up-sample the channels
    Y_u, Cb_u, Cr_u = f.upsampling(
        Y_w, Cb_w, Cr_w, factor, original.shape)

    # Show image with upsampled channels
    if show:
        plt.figure("Upsampled channels")
        # upsampled = f.merge_RGB(Y_u, Cb_u, Cr_u)
        f.visualize_YCbCr(Y_u, Cb_u, Cr_u, "winter")
        plt.show()

    # Transforma de volta RGB
    R, G, B = f.YCbCr_to_RGB(
        Y_u, Cb_u, Cr_u)

    # Visualiza imagem com canais separados
    if show:
        f.visualize_RGB(R, G, B, "gray")

    image_padded = f.merge_RGB(R, G, B)

    # Faz unpadding
    unpadded = f.unpadding(image_padded, original)

    return


def main():
    show = True

    factor = [4, 2, 0]

    # Read image (3.1)
    image_path = "imagens\\peppers.bmp"
    image = f.plt.imread(image_path)

    # Create a colormap (3.2)
    cmUser = f.colormap('cmUser', [(0, 0, 0), (1, 1, 1)], 256)

    # Visualize image with a colormap (3.3)
    # f.image_colormap(image, cmUser)

    blocks64 = encoder(
        image, factor, cmUser, show)

    decoder(blocks64, factor, image, show)


if __name__ == '__main__':

    main()
