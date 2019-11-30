"""face recomender system"""
import argparse

import glob
import os
from utils import detect_face, search_img

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="this is face recomendation system")
    PARSER.add_argument("-i", "--image", type=str, default="../input_img/sample.jpg")
    ARGS = PARSER.parse_args()

    detect_face.detect_face(ARGS.image)
    FACE_LIST = glob.glob("../output_img/face_*.jpg")

    for i, face in enumerate(FACE_LIST):
        if os.path.isfile(face):
            detect_face.plt_img(face)
            flag = input("Search image likely the person? [y/N]")
        if flag == "y":
            likely_images_list = search_img.get_likely_image(face)
