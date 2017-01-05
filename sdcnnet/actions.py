import os
import sys
from .lanefinder import LaneFinderPipeline

test_image_folder = "lane_test_images"


def test_images(pipeline, folder=test_image_folder):
    if not os.path.isdir(folder):
        print("Cannot find the test image folder %s" % test_image_folder)
        print("Please be sure that the current folder is the one containing the project")
        sys.exit(1)
    images = os.listdir(folder)

    for image_name in images:
        # out pipeline start from the imagename, and return the filename with the output
        # quite straightforward
        filename = os.path.join(folder, image_name)
        pipeline.new()
        pipeline.load_image(filename)
        output = pipeline.process_pipeline(show_intermediate=False, show_original=False)


from moviepy.editor import VideoFileClip


def process_video(source_name, target_name):
    pipeline = LaneFinderPipeline()

    def process_image(image):
        pipeline.load_image(image)
        result = pipeline.process_pipeline(
            show_intermediate=False, show_mask=False,
            show_original=False, show_final=False
        )
        return result

    clip1 = VideoFileClip(source_name)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(target_name, audio=False)
