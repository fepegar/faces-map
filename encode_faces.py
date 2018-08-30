"""
Script to encode faces into 128D feature vectors
"""

import sys
from typing import List
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from faces_map import encoding
from faces_map.path import ensure_dir
from faces_map.console import print_color, RED


def main():
    description: str = 'Encode faces found in all photos in a directory'

    examples: List[str] = '\n'.join([
            'Examples:',
            'encode_faces ~/Desktop/facebook_photos/ ~/Desktop/encodings.csv',
        ]
    )

    # RawDescriptionHelpFormatter is used to print examples in multiple lines
    parser: ArgumentParser = ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        'input_dir', type=str,
        help='directory that will be searched for JPEG files recursively')
    parser.add_argument(
        'output_csv', type=str,
        help='path to the CSV file where the encodings will be saved')
    arguments = parser.parse_args()

    input_dir: Path = Path(arguments.input_dir).expanduser()
    if not input_dir.is_dir():
        print_color(f'Error: directory "{input_dir}" does not exist', RED)
        sys.exit(1)

    encodings_path: Path = Path(arguments.output_csv).expanduser()
    ensure_dir(encodings_path)

    faces_data = encoding.encode_faces(input_dir)
    faces_data.save(encodings_path)

if __name__ == '__main__':
    main()
