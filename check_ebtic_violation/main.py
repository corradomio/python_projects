import logging.config
from lab_dress_glove import classify_image

def main():

    # clearer
    ret = classify_image(r"test_images\20260123_094717_crop_no_margin.jpg")
    # security
    ret = classify_image(r"test_images\20260607_181414_crop_no_margin.jpg")

    ret = classify_image(r"test_images\20260218_132846_crop_no_margin.jpg")
    ret = classify_image(r"test_images\20260218_133832_crop_no_margin.jpg")

    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    main()