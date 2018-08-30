# Faces Map

## Installation

Set up a new [`conda`](https://conda.io/) environment:

```shell
$ ENV_NAME="faces"
$ conda create -n $ENV_NAME "python>=3.6"
$ source activate $ENV_NAME
```

Install `pip` package:

```shell
(faces)$ pip install faces_map
```

### Optional (for experimental features)
```shell
(faces)$ conda install matplotlib
(faces)$ conda install vtk
```

## Usage
```shell
(faces)$ DIR_WITH_PHOTOS="awesome_photos/"
(faces)$ OUTPUT_CSV ="encodings.csv"
(faces)$ OUTPUT_MAP ="embedding.jpg"
(faces)$ encode_faces $DIR_WITH_PHOTOS $OUTPUT_CSV
(faces)$ embed_faces $OUTPUT_CSV $OUTPUT_MAP
```

If you want to use your Facebook tagged photos, you can download them as explained in the [`download_photos`](download_photos.py) script. For example:

```shell
(faces)$ IN_LOG="facebook.log"
(faces)$ OUT_DIR="fb_photos/"
(faces)$ download_photos $IN_LOG $OUT_DIR
```
