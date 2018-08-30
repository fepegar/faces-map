import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

from .console import print_color, RED, GREEN


def read_log(log_path: Path) -> Dict[str, str]:
    """
    Read a Chrome log file with Facebook URLs.

    First line is expected to be the JavaScript command used to print URLs
    Second line is expected to be blank
    Last is expected to be "undefined"
    """
    lines: List[str] = log_path.read_text().splitlines()
    urls: Dict[str, str] = {}
    for line in lines[2:-1]:  # skip JS command, blank line and "undefined"
        try:
            url: str = line.split()[1]
            image_id: Optional[str] = get_id_from_url(url)
            if image_id is None:
                print_color(f'No JPEG image ID found in URL {url !r}\n', RED)
                continue
            urls[image_id] = url
        except IndexError:
            print_color(f'Error getting URL from {line !r}\n', RED)
    return urls


def get_id_from_url(url: str) -> Optional[str]:
    pattern: str = r'/(\w*).jpg'
    matches: List[str] = re.findall(pattern, url)
    return matches[0] if len(matches) == 1 else None


def download_urls(urls: Dict[str, str], output_dir: Path):
    num_urls: int = len(urls)
    for i, (image_id, url) in enumerate(urls.items()):
        image_path: Path = output_dir / f'{image_id}.jpg'
        print(f'Downloading {i + 1}/{num_urls}: {url}')
        try:
            urllib.request.urlretrieve(url, image_path)
            print_color(f'Saved in {image_path}', GREEN)
        except Exception as e:
            print_color('Error downloading from URL', RED)
            print(e)
        print()
