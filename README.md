# Faces Map
This is a tool that generates an image representing a distribution of faces according to features computed by a neural network.

## How it works
1. Detect faces using [histogram of oriented gradients (HOG)](https://www.learnopencv.com/histogram-of-oriented-gradients/) or a [face-recognition convolutional neural network (CNN)](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)
2. Extract 128 features from each face using a[face-encoding CNN](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)
3. Embed the 128D feature vectors into a 2D space using [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/)
4. Show faces at their corresponding coordinates


## Installation

Set up a new [`conda`](https://conda.io/) environment:

```shell
$ ENV_NAME="faces"
$ conda create -n $ENV_NAME python=3.6
$ source activate $ENV_NAME
```

Install `pip` package:

```shell
(faces) $ pip install faces_map
```

### Optional (for experimental features)
```shell
(faces) $ conda install matplotlib
(faces) $ conda install vtk
```

## Usage
```shell
(faces) $ DIR_WITH_PHOTOS="awesome_photos/"
(faces) $ OUTPUT_CSV ="encodings.csv"
(faces) $ OUTPUT_MAP ="embedding.jpg"
(faces) $ encode_faces $DIR_WITH_PHOTOS $OUTPUT_CSV
(faces) $ embed_faces $OUTPUT_CSV $OUTPUT_MAP
```

If you want to use Facebook photos, you can download them as explained in the [`download_photos`](download_photos.py) script and use [`log2map`](log2map.py) to create the embedding directly. For example, running
```shell
(faces) $ log2map facebook.log
```
generates:

1. A photos directory `facebook/`
2. An encodings file `facebook.csv`
3. An embedding file `facebook.jpg`
