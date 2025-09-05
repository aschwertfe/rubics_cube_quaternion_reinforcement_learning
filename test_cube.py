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

def test_visualization(cube, viewer, K:int):
    if K == 0: 
        time.sleep(10.0)
    else:
        for k in range(K):
            time.sleep(1.0)
            randomized_rotation(cube)
            viewer.update_position()
            print("Server updating")

def randomized_rotation(cube):
    choice = np.random.randint(low=0,high=3,size=2)
    axis, depth = choice[0], choice[1]-1
    dir = +1
    cube.rotate(axis, depth, dir, with_faces=True)

def test_rotation_correctness(cube, K:int):
    for i in range(K):
        randomized_rotation(cube)

    idx, quats = cube.get_positions(quats=True)
    quats = quats.flatten()
    duplicates = quats[np.unique(quats, return_counts=True)[1] > 1]
    print("Duplicates: ", np.unique(duplicates) != [] ) # if list is empty there are no duplicates -> False


if __name__ == "__main__":
    cube = initialize_cube()
    viewer = initialize_server(cube)
    test_visualization(cube, viewer, 30)
    #test_rotation_correctness(cube, 1)
    #test_visualization(cube, viewer, 10)
