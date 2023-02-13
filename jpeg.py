import header as h


def encoder(image_path):

    print('Encoder')


def decoder():

    print('Decoder')


def main():

    # Read image (3.1)
    image_path = "imagens\logo.bmp"
    image = h.plt.imread(image_path)

    # Create a colormap (3.2)
    cmUser = h.colormap('cmUser', [(0, 0, 0), (1, 0, 0)], 256)

    # Visualize image with a colormap (3.3)
    h.image_colormap(image, cmUser)

    # Separate the image in 3 channels (3.4)
    R, G, B = h.separate_RGB(image)

    # Merge the 3 channels into an image (3.4)
    image = h.merge_RGB(R, G, B)

    # Visualize the image and each one of its channels (with the adequate colormap) (3.5)
    h.visualize_RGB(R, G, B)

    # Padding of the image (4.1)
    image = h.padding(image, 2)

    # Convert image from RGB to YCbCr (5.1)
    Y, Cb, Cr = h.RGB_to_YCbCr(image)

    # Convert image from YCbCr to RGB (5.1)
    image = h.YCbCr_to_RGB(Y, Cb, Cr)

    # Visualize the image and its YCbCr components (5.2)
    h.visualize_YCbCr(Y, Cb, Cr)


if __name__ == '__main__':

    main()
