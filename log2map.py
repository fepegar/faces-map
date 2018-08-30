from typing import List
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from faces_map import download, encoding
from faces_map.face import FacesData, FACE_SIZE, FACES_SIDE, PERPLEXITY


def main():
    description: str = 'Download all photos listed in a Facebook console log'

    examples: List[str] = '\n'.join(
        ['Examples:',
         'download_photos facebook.log ~/Desktop/facebook_photos/',
        ])

    # RawDescriptionHelpFormatter is used to print examples in multiple lines
    parser: ArgumentParser = ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        'input_log', type=Path,
        help='path to a file containing the output of the browser console')

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

    log_path: Path = Path(arguments.input_log)
    stem = log_path.stem
    parent = log_path.parent

    # Download images
    photos_dir: Path = parent / f'{stem}'
    if not photos_dir.is_dir():
        download.download_urls(arguments.input_log, photos_dir)

    # Encode faces
    encodings_path: Path = parent / f'{stem}.csv'
    if not encodings_path.is_file():
        encoding.encode_faces(photos_dir, encodings_path=encodings_path)

    # Embed faces
    embedding_path: Path = parent / f'{stem}.jpg'
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
