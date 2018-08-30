"""
Script to generate the embedding image
"""

import sys
from typing import List
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from faces_map.path import ensure_dir
from faces_map.console import print_color, RED
from faces_map.face import FacesData, FACE_SIZE, FACES_SIDE, PERPLEXITY


def main():
    description: str = 'Embed encoded faces in a 2D map and save it'

    examples: List[str] = '\n'.join([
        'Examples:',
        'embed_faces ~/Desktop/encodings.csv ~/Desktop/embedding.jpg',
        ])

    # RawDescriptionHelpFormatter is used to print examples in multiple lines
    parser: ArgumentParser = ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        'input_csv', type=str,
        help='path to the CSV file where the encodings have been saved')
    parser.add_argument(
        'output_jpg', type=str,
        help='path to the image file where the embedded faces will be saved')
    parser.add_argument(
        '--faces-side', '-d', type=int, default=FACES_SIDE,
        help=(
            'number of faces on the side of the embedding image'
            f' default: {FACES_SIDE}'
        )
    )
    parser.add_argument(
        '--face-size', '-z', type=int, default=FACE_SIZE,
        help=(
            'size in pixels of each face in the embedding image'
            f' default: {FACE_SIZE}'
        )
    )
    parser.add_argument(
        '--perplexity', '-p', type=float, default=PERPLEXITY,
        help=(
            'perplexity parameter fot t-SNE algorithm'
            f' default: {PERPLEXITY}'
        )
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='fill each cell with the face with closest coordinates'
    )
    parser.add_argument(
        '--no-overwrite', action='store_true',
        help='do not overwrite encodings file with embedding coordinates'
    )
    arguments = parser.parse_args()

    encodings_path: Path = Path(arguments.input_csv).expanduser()
    if not encodings_path.is_file():
        print_color(f'Error: file "{encodings_path}" does not exist', RED)
        sys.exit(1)

    embedding_path: Path = Path(arguments.output_jpg).expanduser()
    ensure_dir(embedding_path)

    faces_data = FacesData(encodings_path)
    faces_data.make_tsne_montage(
        embedding_path,
        faces_side=arguments.faces_side,
        face_size=arguments.face_size,
        nearest=arguments.nearest
    )

    # Add t-SNE coordinates to faces table
    overwrite_encodings = not arguments.no_overwrite
    if overwrite_encodings:
        print(f'Overwriting {encodings_path}...')
        faces_data.save(encodings_path)
        print('t-SNE coordinates added to', encodings_path)

if __name__ == '__main__':
    main()
