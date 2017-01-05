from .base import *


class LaneFinderPipeline:
    def __init__(self):
        # maybe we will need some params here
        self.image = None
        self.interesting_mask = None  # interest zone mask is the same for each frame
        self.previous_lines = None  # we use the previous lines to average the new ones

    def new(self):
        """ We call new, when we want to apply the pipeline to something unrelated """
        self.interesting_mask = None
        self.previous_lines = None

    def load_image(self, src):
        if isinstance(src, str):
            # print("Loading image from file %s" % src)
            self.image = load_image(src)
        else:
            self.image = src

    def apply_region_of_interest(self, src=None):
        img = src if src is not None else self.image
        if self.interesting_mask is None:
            self.interesting_mask = get_mask(img)
        return cv2.bitwise_and(img, self.interesting_mask)

    def process_pipeline(self,
                         show_intermediate=False, show_mask=False,
                         show_original=True, show_final=True):
        image = self.image
        if image is None:
            raise Exception("Please load an image before starting the process")

        if show_original:
            plt.figure()
            plt.title("Original")
            plt.imshow(image)

        gray = image_to_int(grayscale(image))
        canned = self.apply_region_of_interest(canny(gaussian_blur(gray)))

        # Hough:
        lines = hough_lines(canned)  # get all the lines from canny
        # normal_hug = hough(canned, lines, color=(0, 255, 0))
        if self.previous_lines:  # we add the previous known lanes in the mix
            lines = list(lines) + self.previous_lines
        lines = lane_enhance(lines)  # consider the meaningful lines and average/extend them
        hug = hough(canned, lines=lines)
        self.previous_lines = lines

        # let's build a nice output using some intermediate result
        combo = self.image
        if show_mask:
            mask = get_mask(image, mask_color=(0, 200, 0))
            combo = weighted_img(combo, mask, .2, 1.)
            # combo = weighted_img(combo, normal_hug)
        combo = weighted_img(hug, combo)

        if show_final:
            plt.figure()
            plt.title('Combined')
            plt.imshow(combo)
        return combo


# pipeline = LaneFinderPipeline()
# pipeline.load_image('lane_test_images/solidWhiteRight.jpg')
# result = pipeline.process_pipeline(show_original=False, show_intermediate=False, show_mask=True)
