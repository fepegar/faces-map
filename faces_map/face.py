# https://stackoverflow.com/a/33533514/3956024
from __future__ import annotations
import sys
import string
from pathlib import Path
from itertools import product
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

from .console import print_color, RED

NOT_CLASSIFIED = -1
FACE_SIZE = 60
FACES_SIDE = 70
PERPLEXITY = 30
NUM_FEATURES = 128
PIXEL_SPACING = 0.0001


class BoundingBox:
    def __init__(self, box=None, row=None):
        if row is None:
            self.tuple = box
            self.top, self.right, self.bottom, self.left = self.tuple
        else:
            self.top = row.top
            self.right = row.right
            self.bottom = row.bottom
            self.left = row.left
            self.tuple = self.top, self.right, self.bottom, self.left


    def __repr__(self):
        info = (
            f'Top: {self.top},'
            f' bottom: {self.bottom},'
            f' left: {self.left},'
            f' right: {self.right}'
        )
        return f'{self.__class__.__name__}({info})'

    @property
    def size(self):
        return (self.bottom - self.top) * (self.right - self.left)


    def get_pillow_box(self):
        return self.left, self.top, self.right, self.bottom



class Face:
    def __init__(self, image_path=None, box=None, encoding=None, row=None):
        if row is None:
            self.image_path = image_path
            self.bounding_box = BoundingBox(box=box)
            self.encoding = encoding
        else:
            self.image_path = row.path
            self.bounding_box = BoundingBox(row=row)
            self.encoding = row.filter(regex='feature*').values
            self.label = None
            self.coordinates_2d = None
            self.coordinates_3d = None
            if 'label' in row:
                self.label = row['label']
            if 'x_tsne_2D' in row:
                self.coordinates_2d = np.array(
                    (row['x_tsne_2D'], row['y_tsne_2D']))
            if 'x_tsne_3D' in row:
                self.coordinates_3d = np.array(
                    (row['x_tsne_3D'], row['y_tsne_3D'], row['z_tsne_3D']))


    def __repr__(self):
        info = f'{self.image_path}, {self.bounding_box}'
        return f'{self.__class__.__name__}({info})'


    def get_dict(self):
        face_data = OrderedDict()
        face_data['path'] = self.image_path
        face_data['top'] = self.bounding_box.top
        face_data['right'] = self.bounding_box.right
        face_data['bottom'] = self.bounding_box.bottom
        face_data['left'] = self.bounding_box.left
        for i, value in enumerate(self.encoding):
            face_data[f'feature_{i}'] = value
        return face_data


    def get_array(self, resize=True, face_size=FACE_SIZE):
        return np.array(self.get_image(resize=resize, face_size=face_size))


    def get_image(self, resize=True, face_size=FACE_SIZE):
        image = Image.open(self.image_path)
        box = self.bounding_box.get_pillow_box()
        if resize:
            size = face_size, face_size
            return image.resize(size, resample=Image.BICUBIC, box=box)
        else:
            return image.crop(box)


    def plot(self, axis):
        try:
            from matplotlib import offsetbox  # pylint: disable=import-error
        except ModuleNotFoundError:
            show_import_error()
        roi = self.get_array()
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(roi),
            self.coordinates_2d,
            pad=0)
        axis.add_artist(imagebox)


    def get_actor(self, follow=True):
        from . import vtk_utils
        x, y, z = self.coordinates_3d
        origin = x, y, z
        spacing = 3 * [PIXEL_SPACING]
        poly_data = vtk_utils.array_to_poly_data(
            self.get_array(), spacing=spacing, origin=origin)
        actor = vtk_utils.get_actor_from_poly_data(poly_data, follower=follow)
        return actor



class FacesData:
    def __init__(self, data) -> None:
        if isinstance(data, list):
            faces_list = data
            faces_data = [face.get_dict() for face in faces_list]
            self.df = pd.DataFrame(faces_data)
        elif isinstance(data, (str, Path)):
            csv_path = data
            self.df = pd.read_csv(csv_path, index_col=0)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        self.set_faces_size()


    def __repr__(self):
        return repr(self.df)


    def __len__(self):
        return len(self.df)

    @property
    def X(self):
        return self.df.filter(regex='feature*').values

    @property
    def num_faces(self):
        return len(self.df)

    @property
    def faces(self):
        return [Face(row=row)
                for (_, row) in self.df.iterrows()]


    def save(self, csv_path) -> None:
        self.df.to_csv(csv_path)


    def set_labels(self, labels) -> None:
        self.df['label'] = labels


    def ensure_coordinates(self, dimensions):
        if f'x_tsne_{dimensions}D' not in self.df:
            self.compute_tsne(dimensions)


    def get_coordinates(self, dimensions):
        self.ensure_coordinates(dimensions)
        if dimensions == 2:
            x = self.df['x_tsne_2D'].values
            y = self.df['y_tsne_2D'].values
            coordinates = np.column_stack((x, y))
        elif dimensions == 3:
            x = self.df['x_tsne_3D'].values
            y = self.df['y_tsne_3D'].values
            z = self.df['z_tsne_3D'].values
            coordinates = np.column_stack((x, y, z))
        return coordinates


    def set_coordinates(self, coordinates, dimensions) -> None:
        if dimensions == 2:
            x, y = coordinates.T
            self.df['x_tsne_2D'] = x
            self.df['y_tsne_2D'] = y
        elif dimensions == 3:
            x, y, z = coordinates.T
            self.df['x_tsne_3D'] = x
            self.df['y_tsne_3D'] = y
            self.df['z_tsne_3D'] = z


    def normalize_coordinates(self, dimensions):
        def normalize_array(array):
            np.subtract(array, array.min(), out=array)
            np.divide(array, array.max(), out=array)
        if dimensions == 2:
            x, y = self.get_coordinates(dimensions).T
        elif dimensions == 3:
            x, y, z = self.get_coordinates(dimensions).T
        normalize_array(x)
        normalize_array(y)
        if dimensions == 2:
            normalized_coordinates = np.column_stack((x, y))
        elif dimensions == 3:
            normalize_array(z)
            normalized_coordinates = np.column_stack((x, y, z))
        self.set_coordinates(normalized_coordinates, dimensions)


    def demean_coordinates(self, dimensions):
        coordinates = self.get_coordinates(dimensions)
        coordinates -= coordinates.mean(axis=0)
        self.set_coordinates(coordinates, dimensions)


    def cluster(self, min_samples: int = 3) -> None:
        print('Computing DBSCAN...')
        from sklearn.cluster import DBSCAN
        clt = DBSCAN(min_samples=min_samples)
        clt.fit(self.X)
        self.set_labels(clt.labels_)


    def compute_tsne(self,
                     dimensions: int,
                     perplexity: float = PERPLEXITY) -> None:
        print(f'Computing {dimensions}D t-SNE...')
        from sklearn.manifold import TSNE
        algorithm = TSNE(
            n_components=dimensions, random_state=0, perplexity=perplexity)
        coordinates_tsne = algorithm.fit_transform(self.X)
        self.set_coordinates(coordinates_tsne, dimensions)
        self.normalize_coordinates(dimensions)


    def remove_most_frequent(self) -> None:
        most_frequent_class = self.get_most_frequent_class()
        print('Removing class', most_frequent_class)
        self.df = self.df[self.df.label != most_frequent_class]


    def remove_unclassified(self) -> None:
        self.df = self.df[self.df.label >= 0]


    def get_most_frequent_class(self):
        count = self.df.label.value_counts()
        count_index = 0 if count.index[0] != NOT_CLASSIFIED else 1
        most_frequent_index = count.index[count_index]
        return most_frequent_index


    def convert_labels_to_letters(self):
        indices = self.df.label.values + 1
        letters_list = string.ascii_uppercase + string.ascii_lowercase
        letters = [letters_list[i] for i in indices]
        self.df.label = pd.Categorical(letters)


    def plot(self,
             faces: bool = True,
             remove_most_frequent: bool = False,
             remove_unclassified: bool = True,
             sort_faces: bool = True) -> None:
        try:
            import matplotlib.pyplot as plt  # pylint: disable=import-error
        except ModuleNotFoundError:
            show_import_error()
        copy = self.copy()
        if remove_most_frequent:
            copy.remove_most_frequent()
        if remove_unclassified:
            copy.remove_unclassified()
        copy.convert_labels_to_letters()

        _, ax = plt.subplots()
        try:
            import seaborn as sns  # pylint: disable=import-error
        except ModuleNotFoundError:
            show_import_error(module='seaborn', manager='pip')
        sns.set()
        sns.scatterplot(x='x_tsne_2D', y='y_tsne_2D', data=copy.df,
                        legend=False, edgecolor='', s=10,
                        hue='label', alpha=0.5, ax=ax)
        if faces:
            if sort_faces:
                copy.sort_by_size()
            copy.plot_faces(ax)
            plt.axis('off')
        plt.show()


    def plot_faces(self, axis):
        for i, (_, row) in enumerate(self.df.iterrows()):
            print(f'Plotting face {i + 1}/{len(self.df)}')
            face = Face(row=row)
            face.plot(axis=axis)


    def copy(self) -> FacesData:
        return FacesData(self.df.copy())


    def set_faces_size(self) -> None:
        size = (self.df.bottom - self.df.top) * (self.df.right - self.df.left)
        self.df['face_size'] = size


    def sort_by_size(self, ascending: bool = True) -> None:
        self.set_faces_size()  # in case it hasn't been done before
        self.df = self.df.sort_values(by='face_size', ascending=ascending)


    def make_features_montages(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        for feature_idx in range(NUM_FEATURES):
            print(f'Computing feature {feature_idx + 1} / {NUM_FEATURES}')
            basename = f'feature_{feature_idx}'
            filename = f'{basename}.jpg'
            filepath = output_dir / filename
            copy = self.copy()
            copy.df = copy.df.sort_values(by=basename)
            faces_side = copy.make_square()
            pixels_side = FACE_SIZE * faces_side
            output_shape = pixels_side, pixels_side, 3
            output_array = np.empty(output_shape, np.uint8)

            for row_idx, (_, row) in enumerate(copy.df.iterrows()):
                print(f'Computing face {row_idx + 1} / {copy.num_faces}')
                face = Face(row=row)
                i = row_idx // faces_side
                j = row_idx % faces_side
                i_ini = i * FACE_SIZE
                j_ini = j * FACE_SIZE
                i_fin = (i + 1) * FACE_SIZE
                j_fin = (j + 1) * FACE_SIZE
                output_array[i_ini:i_fin, j_ini:j_fin] = face.get_array()
            image = Image.fromarray(output_array)
            image.save(filepath)


    def make_square(self):
        """
        Remove faces from the middle of the sorted data frame so that a square
        montage can be generated
        """
        faces_side = int(np.floor(np.sqrt(self.num_faces)))
        num_faces_to_remove = self.num_faces - faces_side**2
        middle_index = len(self) // 2
        remove_ini = middle_index - num_faces_to_remove // 2
        remove_fin = remove_ini + num_faces_to_remove
        self.df = self.df.drop(self.df.index[remove_ini:remove_fin])
        return faces_side


    def make_tsne_montage(self,
                          output_path,
                          faces_side=FACES_SIDE,
                          face_size=FACE_SIZE,
                          nearest=False):
        self.ensure_coordinates(dimensions=2)
        if nearest:
            self.make_tsne_nearest_montage(
                output_path, faces_side=faces_side, face_size=face_size)
        else:
            self.make_tsne_exact_montage(
                output_path, faces_side=faces_side, face_size=face_size)


    def make_tsne_exact_montage(self, output_path,
                                faces_side=FACES_SIDE, face_size=FACE_SIZE):
        self.sort_by_size(ascending=False)  # good quality first
        taken_cells = []
        side_pixels = faces_side * face_size
        shape = side_pixels, side_pixels, 3
        output_array = np.zeros(shape, np.uint8)
        for face_idx, face in enumerate(self.faces):
            print(f'Processing face {face_idx + 1}/{self.num_faces}')
            cell = np.uint16(np.round(face.coordinates_2d * (faces_side - 1)))
            cell = tuple(cell)  # tuples are hashable
            if cell in taken_cells:
                continue
            else:
                taken_cells.append(cell)
            j, i = cell
            i_ini = i * face_size
            j_ini = j * face_size
            i_fin = (i + 1) * face_size
            j_fin = (j + 1) * face_size
            face_array = face.get_array(face_size=face_size)
            output_array[i_ini:i_fin, j_ini:j_fin] = face_array
        image = Image.fromarray(output_array)
        image.save(output_path)
        print('Montage written to', output_path)


    def make_tsne_nearest_montage(self, output_path,
                                  faces_side=FACES_SIDE, face_size=FACE_SIZE):
        dimensions = 2
        side_pixels = faces_side * face_size
        shape = side_pixels, side_pixels, 3
        output_array = np.zeros(shape, np.uint8)
        all_grid_coordinates = product(range(faces_side), repeat=2)
        faces_coordinates = self.get_coordinates(dimensions)
        for grid_coordinates in all_grid_coordinates:
            grid_coordinates = np.array(grid_coordinates)
            grid_coordinates_norm = grid_coordinates / (faces_side - 1)
            grid_xy_norm = np.flip(grid_coordinates_norm)
            diffs = faces_coordinates - grid_xy_norm
            distances = np.linalg.norm(diffs, axis=1)
            closest_index = np.argmin(distances)
            face = Face(row=self.df.iloc[closest_index, :])
            i, j = grid_coordinates
            i_ini = i * face_size
            j_ini = j * face_size
            i_fin = (i + 1) * face_size
            j_fin = (j + 1) * face_size
            face_array = face.get_array(face_size=face_size)
            output_array[i_ini:i_fin, j_ini:j_fin] = face_array
        image = Image.fromarray(output_array)
        image.save(output_path)
        print('Montage written to', output_path)


    def make_tsne_montages(self, output_dir,
                           faces_numbers=None, nearest=False):
        if faces_numbers is None:
            faces_numbers = range(10, 110, 10)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        for n in faces_numbers:
            filename = f'tsne_{n}_faces.jpg'
            tsne_montage_path = output_dir / filename
            self.make_tsne_montage(
                tsne_montage_path, faces_side=n, nearest=nearest)


    def show_tsne_3d(self, follow=True):
        from . import vtk_utils
        print(f'Getting {self.num_faces} faces actors...')
        self.demean_coordinates(dimensions=3)
        actors = [face.get_actor(follow=follow) for face in self.faces]
        vtk_utils.show_scene(actors)


def show_import_error(module='matplotlib', manager='conda'):
    print_color(f'{module} not found. '
                f'Install with "{manager} install {module}"',
                RED)
    sys.exit(1)
