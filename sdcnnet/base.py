# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from operator import itemgetter
from functools import reduce


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size=9):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def image_default_vertices(image):
    height, width = image.shape[:2]
    # return a trapezoind on bottom image to around the center of image

    # [(0, 540), (384.0, 270.0), (576.0, 270.0), (960, 540)]
    vertices = [(- width * 0.1, height),
                (int(width * 0.45), height * 0.6),
                (int(width * 0.55), height * 0.6),
                (width * 1.1, height)]
    return np.array([vertices],
                    dtype=np.int32)


def get_mask(img, vertices=None, mask_color=None):
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    if vertices is None:
        vertices = image_default_vertices(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if mask_color is None:
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            mask_color = (255,) * channel_count
        else:
            mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, mask_color)
    return mask


def region_of_interest(img, vertices=None):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # returning the image only where mask pixels are nonzero
    mask = get_mask(img, vertices)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=None, thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if color is None:
        color = [255, 0, 0]
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    else:
        print("No lines found")


def hough_lines(img, rho=2, theta=np.pi / 180, threshold=15, min_line_len=30, max_line_gap=20):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def hough(img, lines=None, **kwargs):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    if lines is None:
        lines = hough_lines(img, **kwargs)
    color = kwargs.pop('color', None)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, color=color)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=1., β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def load_image(filename):
    image = mpimg.imread(filename)
    return image


def image_to_int(image):
    if image.dtype != 'uint8':
        return (image * 255).astype('uint8')
    else:
        return image


def save_image(img, filename):
    mpimg.imsave(filename, img)

def lane_enhance(lines):
    results = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) < 0.4:  # ignore ~ horizontal lines
                    continue
                results.append(line)
    else:
        print("No lines found")
    # let's extend all the lines, and then average them scoring on the original lenght
    extended = extend_lines(results, with_scores=True)

    return line_averages(extended, with_scores=True)

def extend_lines(lines, with_scores=False):
    """ Return the same array of lines, but extended from minY to maxY
        if with_scores == True it returns a tuple (line, score) scoring with line length
    """
    miny = maxy = None
    for line in lines:
        for x1, y1, x2, y2 in line:
            miny = min(y1, y2) if miny is None else min(miny, y1, y2)
            maxy = max(y1, y2) if maxy is None else max(maxy, y1, y2)
    if miny is not None:
        miny = max(miny, maxy/2) # to avoid too long lane lines

    results = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            q = y1 - slope * x1
            x_at_maxy = (maxy - q) / slope
            x_at_miny = (miny - q) / slope
            line = (x_at_miny, miny, x_at_maxy, maxy)
            if with_scores:
                score = (x2 - x1) ** 2 + (y2 - y1) ** 2  # the linelen^2
                results.append([[line], score])
            else:
                results.append([line])
    return results


def line_averages(lines, with_scores=False):
    """ Given a bunch of lines return two lines with that are an 'average' of all the line
    grouping with the sign of slope.
    Every line can be a tuple (line, score) that will be used for weighted averages
    """
    if with_scores is False:
        lines = [(line, 1) for line in lines]

    positive_slopes = []
    negative_slopes = []
    # print("average between %d lines" % len(lines))
    for line, score in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope > 0:
            positive_slopes.append((line, score))
        else:
            negative_slopes.append((line, score))

    results = []
    def do_averages(lines):
        """ Given all lines and scores, return a single line that is the weighted avg """
        if lines is None:
            return None
        total_scores = sum(score for line, score in lines)
        # print("total score", total_scores)
        avgx1 = avgy1 = avgx2 = avgy2 = 0
        for line, score in lines:
            x1, y1, x2, y2 = line[0]
            avgx1 += x1 * score
            avgy1 += y1 * score
            avgx2 += x2 * score
            avgy2 += y2 * score
        return [(avgx1 / total_scores, avgy1 / total_scores,
                 avgx2 / total_scores, avgy2 / total_scores)]

    if positive_slopes:
        results.append(do_averages(positive_slopes))
    if negative_slopes:
        results.append(do_averages(negative_slopes))
    return results

