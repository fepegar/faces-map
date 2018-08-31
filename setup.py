from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='faces_map',
    version='0.1.0',
    author='Fernando Perez-Garcia',
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fepegar/faces-map",
    packages=find_packages(exclude=['*tests']),
    entry_points={
        'console_scripts': [
            'download_photos = download_photos:main',
            'encode_faces = encode_faces:main',
            'embed_faces = embed_faces:main',
            'log2map = log2map:main',
        ]
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'LICENSE :: OSI APPROVED :: GNU GENERAL PUBLIC LICENSE (GPL)',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'face_recognition',
        'Pillow',
        'scikit-learn',
        'pandas',
        'scipy',
        ],
    )
