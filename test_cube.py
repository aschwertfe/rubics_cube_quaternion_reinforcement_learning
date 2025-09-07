import utils.cube_env as env
import utils.view as view
import numpy as np
import time

def initialize_cube():
    cube = env.RubicsCube()
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
    cube.disorder(1)
    # choice = np.random.randint(low=0,high=3,size=2)
    # axis, depth = choice[0], choice[1]
    # dir = +1
    # cube.rotate(axis, depth, dir, with_faces=True)

def test_rotation_correctness(cube, K:int):

    for i in range(K):
        randomized_rotation(cube)

    idx, quats = cube.get_positions(quats=True)

    quats = quats.flatten()
    duplicates = quats[np.unique(quats, return_counts=True)[1] > 1]
    print("Duplicates: ", np.unique(duplicates) != [] ) # if list is empty there are no duplicates -> False

def test_correct_rewarding(cube, viewer, with_visualization=True):

    if 10 > cube.get_reward():
            print("Terminal state gives no big reward! Error!")

    for k in range(10):
        choice = np.random.randint(low=0,high=3,size=1)
        axis = choice[0]
        print(f"Check the turn around axis {axis}")
        dir = +1

        depths = np.array([0,1,2])
        np.random.shuffle(depths)
        for depth in depths[:2]:
            cube.rotate(axis, depth, dir, with_faces=with_visualization)
            viewer.update_position()
            if 10 > cube.get_reward()[1]:
                print("Correct")
            else:
                print("Cube detected as finished even though it is not")
            if with_visualization:
                time.sleep(1.0)

        cube.rotate(axis, depths[-1], dir, with_faces=with_visualization)
        viewer.update_position()
        if 10 > cube.get_reward()[1]:
            print("Cube detected as unfinished even though it is")
            print("Perspective change has effect on reward! Error!")
        else:
            print("Correct")
        if with_visualization:
            time.sleep(1.0)


def test_steps(cube, turns):
    
    for turn in range(turns):
        action = np.random.randint(low=0,high=18,size=1)
        print("Take action: ",action)
        state, reward, done, _, _ = cube.step(action)
        print("Reward: ", reward)
        print("Done: ", done)
        print("State: \n", state)

def test_action_translation():
    for k in range(18):
        print(cube._decodeAction(k))

if __name__ == "__main__":
    cube = initialize_cube()
    viewer = initialize_server(cube)

    #test_visualization(cube, viewer, 30)
    #test_rotation_correctness(cube, 1)
    test_visualization(cube, viewer, 10)

    #test_correct_rewarding(cube, viewer True)

    test_action_translation()

    test_steps(cube,10)
    
