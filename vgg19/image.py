import argparse
import logging
from typing import Optional

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S",
    level=logging.INFO)


def load_and_resize_image(file_name: str, debug: bool = False) -> Optional[np.ndarray]:
    """
    Read an image file and resize the image to (3,224,224).

    Parameters
    ----------
    file_name: str
        input path
    debug: bool
        write rescaled image to disk or not

    Returns
    -------
    ndarray:
        numpy array of rescaled image.
    """

    logging.info(f"Reading image: {file_name}")
    img = cv2.imread(file_name)

    # Handle read error:
    if img is None:
        logging.error(f"Image {file_name} not found")
        return

    logging.info(f"Original image shape: {img.shape}")
    dim = (224, 224)
    logging.info(f"Resizing image to: {dim}")
    resized = cv2.resize(img, dim)

    # Write image to file if debug is on
    if debug:
        output_file_name = file_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        logging.info(f"Writing image to file: {output_file_name}")
        cv2.imwrite(f"{output_file_name}_resized.jpg", resized)

    return np.moveaxis(resized, 2, 0)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Image utils")
    parser.add_argument("file_name", help="Name of file to process")
    args = parser.parse_args()

    # Resize input image (debug will be always on if called this way)
    im = load_and_resize_image(args.file_name, debug=True)

    logging.info(f"New image shape: {im.shape}")