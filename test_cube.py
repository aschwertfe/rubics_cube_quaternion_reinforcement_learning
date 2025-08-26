import cube_env as env
import view
import numpy as np
import time

def initialize_cube():
    cube = env.Rubics_Cube()
    return cube

def initialize_server(cube):
    viewer = view.Visualizer(cube)
    return viewer

def test_visualization(viewer):
    while True:
        time.sleep(5.0)
        print("Server updating")
        viewer.update_position()

def test_rotation_correctness(cube):
    for i in range(1000):
        choice = np.random.randint(low=0,high=3,size=2)
        axis, depth = choice[0], choice[1]-1
        dir = +1
        cube.rotate_option1(axis, depth, dir)

    idx, quats = cube.get_positions(quats="True")
    quats = quats.flatten()
    duplicates = quats[np.unique(quats, return_counts=True)[1] > 1]
    print("Duplicates:", np.unique(duplicates))


if __name__ == "__main__":
    cube = initialize_cube()
    viewer = initialize_server(cube)
    test_rotation_correctness(cube)
    test_visualization(viewer)
