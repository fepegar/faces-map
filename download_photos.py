"""
Adapted from:
https://gnmerritt.net/deletefacebook/2018/04/03/fb-photos-of-me/

Photos URL:
https://www.facebook.com/search/[your_facebook_id]/photos-of/intersect

To get your Facebook ID:
https://zerohacks.com/find-facebook-id/

JavaScript snippet to be pasted in Chrome console (opened with Cmd+alt+J):
for (img of document.getElementsByTagName('img')) { if (!img.alt) continue; console.log(img.src); }

Then click on the console output and save the log with Save as...
"""

from typing import List
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from faces_map import download


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
        'input_log', type=str,
        help='path to a file containing the output of the browser console')
    parser.add_argument(
        'output_dir', type=str,
        help='directory where the photos will be saved')
    arguments = parser.parse_args()

    download.download_urls(arguments.input_log, arguments.output_dir)


if __name__ == '__main__':
    main()
