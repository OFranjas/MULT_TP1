import functions as f


def encoder(image_path):

    print('Encoder')


def decoder():

    print('Decoder')


def main():

    # Read image (3.1)
    image_path = "imagens\logo.bmp"
    image = f.plt.imread(image_path)

    # Create a colormap (3.2)
    cmUser = f.colormap('cmUser', [(0, 0, 0), (1, 1, 1)], 256)

    # Visualize image with a colormap (3.3)
    f.image_colormap(image, cmUser)

    # Separate the image in 3 channels (3.4)
    R, G, B = f.separate_RGB(image)

    # Merge the 3 channels into an image (3.4)
    image = f.merge_RGB(R, G, B)

    # Visualize the image and each one of its channels (with the adequate colormap) (3.5)
    f.visualize_RGB(R, G, B)

    # Padding of the image (4.1)
    image = f.padding(image, 2)

    # Convert image from RGB to YCbCr (5.1)
    image = f.RGB_to_YCbCr(image)

    # Convert image from YCbCr to RGB (5.1)
    image = f.YCbCr_to_RGB(image)

    # Visualize the image and its YCbCr components (5.2)
    f.visualize_YCbCr(image)


if __name__ == '__main__':

    main()
