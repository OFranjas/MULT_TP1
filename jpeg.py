import functions as f


def encoder(image):

    # Separate the image in 3 channels (3.4)
    R, G, B = f.separate_RGB(image)

    # Visualize the image and each one of its channels (with the adequate colormap) (3.5)
    f.visualize_RGB(R, G, B)

    # Padding of the image (4.1)
    image_padded = f.padding(image)

    # Convert image from RGB to YCbCr (5.1)
    Y, Cb, Cr = f.RGB_to_YCbCr(R, G, B)

    # Junta aqui porque não estava a dar dentro da função...
    image_y = f.merge_RGB(Y, Cb, Cr)

    f.visualize_YCbCr(Y, Cb, Cr, image)

    return Y, Cb, Cr, image_padded


def decoder(Y, Cb, Cr, image_padded, original):

    # Transforma de volta RGB
    R, G, B = f.YCbCr_to_RGB(Y, Cb, Cr)

    # Visualiza imagem com canais separados
    f.visualize_RGB(R, G, B)

    # Faz unpadding
    unpadded = f.unpadding(image_padded, original)

    # Junta canais
    image = f.merge_RGB(R, G, B)

    return unpadded, image


def main():

    # Read image (3.1)
    image_path = "imagens\\barn_mountains.bmp"
    image = f.plt.imread(image_path)

    # Create a colormap (3.2)
    cmUser = f.colormap('cmUser', [(0, 0, 0), (1, 1, 1)], 256)

    # Visualize image with a colormap (3.3)
    f.image_colormap(image, cmUser)

    Y, Cb, Cr, padded = encoder(image)

    # decoder(Y, Cb, Cr, padded, image)


if __name__ == '__main__':

    main()
