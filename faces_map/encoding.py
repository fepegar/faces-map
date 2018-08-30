"""
Module used to encode faces into 128D feature vectors
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from face_recognition import face_locations, face_encodings

from .face import Face, FacesData
from .console import print_color, GREEN, YELLOW, RED, BLUE

HOG: str = 'hog'  # histogram of oriented gradients
CNN: str = 'cnn'  # convolutional neural network
detection_methods: set = {HOG, CNN}

Box = Tuple[int, int, int, int]


def encode_faces(dataset_dir: Path, detection_method: str = HOG,
                 encodings_path: Path = None) -> FacesData:
    faces: List[Face] = []
    image_paths: List[Path] = sorted(list(dataset_dir.glob('**/*.jpg')))
    num_paths: int = len(image_paths)
    print_color(f'{num_paths} JPEG files found', BLUE)
    total_faces: int = 0
    for i, image_path in enumerate(image_paths):
        print(f'Processing image {i + 1}/{num_paths}: {image_path}')
        try:
            image: Image = Image.open(image_path)
            array: np.array = np.array(image)
            boxes: List[Box] = face_locations(array, model=detection_method)
            num_faces: int = len(boxes)
            total_faces += num_faces
            if num_faces == 0:
                print_color('No faces found', YELLOW)
            elif num_faces == 1:
                print_color('1 face found', GREEN)
            else:
                print_color(f'{num_faces} faces found', GREEN)
            print()
            encodings: list = face_encodings(array, boxes)
            for box, encoding in zip(boxes, encodings):
                face: Face = Face(image_path=image_path, box=box, encoding=encoding)
                faces.append(face)
        except Exception as e:
            print_color(f'Error processing "{image_path}". Skipping...', RED)
            print(e)

    print_color(
        f'{total_faces} faces found in {num_paths} photos'
        f' ({total_faces / num_paths :.1f} faces per photo)',
        BLUE)

    faces_data: FacesData = FacesData(faces)
    if encodings_path is not None:
        faces_data.save(encodings_path)
    return faces_data
