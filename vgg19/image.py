import argparse
import logging

import cv2

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    level=logging.INFO)


def load_and_resize_image(fname, debug=False):
    logging.info(f"Reading image: {fname}")
    img = cv2.imread(fname)

    # Handle read error:
    if img is None:
        logging.error(f"Image {fname} not found")
        return

    logging.info(f"Original image shape: {img.shape}")
    dim = (224, 224)
    logging.info(f"Resizing image to: {dim}")
    resized = cv2.resize(img, dim)

    # Write image to file if debug is on
    if debug:
        output_fname = fname.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        logging.info(f"Writing image to file: {output_fname}")
        cv2.imwrite(f"{output_fname}_resized.jpg", resized)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Image utils")
    parser.add_argument("fname", help="Name of file to process")
    args = parser.parse_args()

    # Resize input image (debug will be always on if called this way)
    load_and_resize_image(args.fname, debug=True)
