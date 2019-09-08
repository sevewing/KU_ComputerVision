# <!--------------------------------------------------------------------------->
# <!--                    KU - University of Copenhagen                      -->
# <!--                          Faculty of Science                           -->
# <!--                Vision and Image Processing (VIP) Course               -->
# <!-- File       : main.py                                                  -->
# <!-- Description: Filtering and edge detection                             -->
# <!-- Author     : Weisi Li (email missing)                                 -->
# <!--------------------------------------------------------------------------->

__version__ = "$Revision: 2019090801 $"

####################################################################################
import cv2
import scipy.ndimage as scn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from functools import partial


####################################################################################

def normalization(image):
    """
    Normalize an image./ Remap to [0,1]
    Remove the impact of illumination
    :param image:
    :return:
    """
    return image / 255.0

def to_uint8(data):
    """
    This function convert the vector data to unsigned integer 8-bits.
    """
    # Maximum pixel.
    latch = np.zeros_like(data)
    latch[:] = 255

    # Minimum pixel.
    zeros = np.zeros_like(data)

    # Unrolled to illustrate steps.
    d = np.maximum(zeros, data)
    d = np.minimum(latch, d)

    # Cast to uint8.
    return np.asarray(d, np.uint8)

def show_images(images, wtitle="Figure"):
    """
    Show images
    :return:
    """
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a new matplotlib window.
    plt.figure(wtitle)

    # Set the default colormap to gray and apply to current image if any.
    plt.gray()

    # Enumerate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Show the image in a new subplot.
        plt.subplot(2, len(images) / 2, pos + 1)
        plt.title(name)
        plt.imshow(image)

    # Show the images.
    plt.show()

def show_signals(row, images):
    """
    Show multiple signals using matplotlib.
    """
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a list of colors.
    colors = ["C0", "C1", "C2", "C3"]

    # Enumerate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Get the image width and the current color.
        w = image.shape[1]
        color = colors[pos % round(len(images) / 2)]

        # This vector contains a list of all valid grayscale level.
        bins = np.array(range(w))

        # Plot the signal in a new subplot.
        ax[pos].cla()
        ax[pos].set_title(name)
        ax[pos].grid(None, "major", "both")
        ax[pos].axis([0, w, 0, 255])
        ax[pos].plot(bins, image[row, :], color=color, linewidth=1)

    # Show the input image.
    # color = cv2.cvtColor(images["Sigma=1"], cv2.COLOR_GRAY2RGB)
    # cv2.line(color, (0, row), (color.shape[1], row), (255, 255, 255), 2)
    # ax[pos+1].imshow(color)

def update_row(slider, val):
    """
    This function will be performed when the user changes the row slider.
    """
    # Create a mask for the slider to set only integer number.
    slider.val = int(round(val))
    slider.poly.xy[2] = slider.val, 1
    slider.poly.xy[3] = slider.val, 0
    slider.valtext.set_text(slider.valfmt % slider.val)

    # Draw the filtered signal.
    show_signals(slider.val, gaussian_filtered_images)

def gaussian_filter(image, s=8, k=0):
    """
    Gaussian filtering
    :param image: input image
    :param s: sigma
    :param k: kernel size
    :return: filtered image
    """
    filtered = image.copy()

    # Gaussian Blur
    filtered = cv2.GaussianBlur(filtered, (k, k), s)
    return filtered


def gaussian_gradient_magnitude_filter(image, s=8, mode='nearest'):
    """
    Gradient magnitude computation using Gaussian derivatives.
    :param image: input image
    :param s: sigma
    :param mode: mode in gaussian_gradient_magnitude
    :return: filtered image
    """
    filtered = image.copy()
    filtered = normalization(filtered)
    filtered = scn.filters.gaussian_gradient_magnitude(filtered, s, mode=mode)
    return filtered


def laplacian_gaussian_filter(image, s=8):
    """
    Laplacian-Gaussian- (= Mexican hat-) filtering.
    :param image: input image
    :param s: sigma
    :return: filtered image
    """
    filtered = image.copy()
    filtered = normalization(filtered)
    filtered = scn.filters.gaussian_laplace(filtered, s)
    return filtered


def canny_edge(image):
    """
    Canny (or similar) edge detection.
    """
    max = 200
    min = 100
    def on_change_max(value):
        nonlocal max
        max= value

    def on_change_min(value):
        nonlocal min
        min = value

    filtered = image.copy()
    # filtered = to_uint8(normalization(filtered))
    cv2.namedWindow("Canny Threshold")
    cv2.createTrackbar("Max Value", "Canny Threshold", max, 500, on_change_max)
    cv2.createTrackbar("Min Value", "Canny Threshold", min, 200, on_change_min)

    while True:
        edges = cv2.Canny(filtered, min, max, L2gradient=True)
        cv2.imshow('Canny Edge Detection', edges)
        key = cv2.waitKey(1)
        if key == ord("q"):  # when pressing q to exit
            break

    cv2.destroyAllWindows()


# <!--------------------------------------------------------------------------->
# <!--                              MAIN                                     -->
# <!--------------------------------------------------------------------------->
if __name__ == '__main__':
    # Load from Input image filename.
    filename = "./inputs/lenna.jpg"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Init a test sigma list
    sigma_list = {"Sigma=1": 1, "Sigma=2": 2, "Sigma=4": 4, "Sigma=8": 8}

    # Task1: Gaussian Filtering
    gaussian_filtered_images = {k: gaussian_filter(image, v) for k, v in sigma_list.items()}
    show_images(gaussian_filtered_images, "Gaussian Filtering")

    # Task2: Gradient Magnitude
    # gradient_magnitude_images = {k: gaussian_gradient_magnitude_filter(image, v) for k, v in sigma_list.items()}
    # show_images(gradient_magnitude_images, "Gradient Magnitude")

    # Task3: Laplacian Gaussian
    # laplacian_gaussian_images = {k: laplacian_gaussian_filter(image, v) for k, v in sigma_list.items()}
    # show_images(laplacian_gaussian_images, "Laplacian Gaussian")

    # Task4: Canny Edge Detection
    # canny_edge(image)

    # <!--------------------------------------------------------------------------->
    # <!--                  SHOW NOISE AS A 1D IMPULSE FUNCTION                  -->
    # <!--------------------------------------------------------------------------->

    # Image resolution
    h, w = image.shape

    # Create a new matplotlib window.
    fig = plt.figure()
    ax = [plt.subplot(1, 4, x + 1) for x in range(4)]

    # Define the row slider.
    axcolor = "lightgoldenrodyellow"
    slider_ax = plt.axes([0.1225, 0.02, 0.78, 0.03], facecolor=axcolor)
    slider_row = Slider(slider_ax, "Row", 1.0, h, valinit=1, valfmt="%i")
    slider_row.on_changed(partial(update_row, slider_row))

    # Show the matplotlib window.
    show_signals(1, gaussian_filtered_images)

    # Show the images.
    plt.show()