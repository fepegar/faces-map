from tempfile import NamedTemporaryFile
import vtk
from skimage import io


def show_scene(actors, show_unit_cube=False):
    # Create the Renderer, RenderWindow, and RenderWindowInteractor.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    axes = vtk.vtkAxes()
    axes.SetOrigin(0, 0, 0)
    axesMapper = vtk.vtkPolyDataMapper()
    axesMapper.SetInputConnection(axes.GetOutputPort())
    axesActor = vtk.vtkActor()
    axesActor.SetMapper(axesMapper)
    ren.AddActor(axesActor)

    # Add the actors to the renderer.
    print('Adding actors to renderer...')
    for actor in actors:
        ren.AddActor(actor)
        # Set the camera of the followers
        if actor.IsA('vtkFollower'):
            actor.SetCamera(ren.GetActiveCamera())

    if show_unit_cube:
        ren.AddActor(get_actor_from_poly_data(get_cube_poly_data()))

    # Zoom in closer.
    # ren.ResetCamera()
    # ren.GetActiveCamera().Zoom(1.6)

    # Reset the clipping range of the camera; render.
    # ren.ResetCameraClippingRange()

    iren.Initialize()
    renWin.Render()

    print('Starting interactor...')
    iren.Start()


def get_cube_poly_data(length=1):
    cube = vtk.vtkCubeSource()
    cube.SetXLength(length)
    cube.SetYLength(length)
    cube.SetZLength(length)
    cube.Update()
    return cube.GetOutput()


def get_actor_from_poly_data(poly_data, follower=False):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtk.vtkFollower() if follower else vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def array_to_poly_data(array, spacing=None, origin=None):
    """
    TODO: figure out how to create image data directly from array
    TODO: show faces as cubes with texture
    """
    with NamedTemporaryFile(suffix='.jpg') as f:
        filepath = f.name
        io.imsave(filepath, array)
        jpeg_reader = vtk.vtkJPEGReader()
        jpeg_reader.SetFileName(filepath)
        jpeg_reader.Update()
    image_data = jpeg_reader.GetOutput()
    if spacing is not None:
        image_data.SetSpacing(spacing)
    if origin is not None:
        image_data.SetOrigin(origin)
    geometry = vtk.vtkImageDataGeometryFilter()
    geometry.SetInputData(image_data)
    geometry.Update()
    poly_data = geometry.GetOutput()
    return poly_data
