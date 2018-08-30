from setuptools import setup, find_packages

setup(
    name='faces_map',
    version='0.1.0',
    author='Fernando Perez-Garcia',
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    packages=find_packages(exclude=['*tests']),
    entry_points={
        'console_scripts': [
            'download_photos = download_photos:main',
            'encode_faces = encode_faces:main',
            'embed_faces = embed_faces:main',
        ]
    },
    install_requires=[
        'face_recognition',
        'Pillow',
        'scikit-learn',
        'pandas',
        ],
    )
