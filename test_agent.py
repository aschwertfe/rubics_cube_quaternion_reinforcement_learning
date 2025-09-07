import utils.cube_env as env
import utils.dqn_agent as ag
import utils.view as view
import time
import os

if __name__ == "__main__":

    ################## Training #################
    
    agent = ag.RLAgent()

    agent.train(epochs=2)

    #agent.save("rubics_dqn_checkpoint_1")

    ########## Loading and evaluating ##########

    # agent = ag.RLAgent()
    # path = os.path.abspath("rubics_dqn_checkpoint_1")
    # agent.load(path)

    cube = env.RubicsCube(mode="quat")
    viewer = view.Visualizer(cube)
    cube.disorder_turns = 1

    for k in range(10):

        state = cube.reset()
        viewer.update_position()
        time.sleep(1.0)

        action = agent.act(state)

        state, reward, done, _ = cube.step(action)
        viewer.update_position()
        print(f"Iter {k} with reward: {reward}")

        time.sleep(1.0)
    
