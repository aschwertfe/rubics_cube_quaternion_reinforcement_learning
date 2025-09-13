import utils.cube_env as env
import utils.dqn_agent as ag
import utils.view as view
import time
import os

if __name__ == "__main__":

    disorder_turns = 2
    max_steps = 2
    epochs = 20
    lesson = 2

    ################## Training #################
    
    agent = ag.RLAgent(disorder_capability=disorder_turns, max_steps= max_steps)

    agent.train(lesson = lesson, epochs=20)

    ########## Loading and evaluating ##########

    # agent = ag.RLAgent()
    path = os.path.join(os.path.abspath('.'), f'data/lesson_{lesson}_checkpoint_{epochs-1}')
    agent.load(path)

    cube = env.RubicsCube(mode="quat",disorder_turns=disorder_turns, max_steps=max_steps)
    viewer = view.Visualizer(cube)
    cube.disorder_turns = disorder_turns

    for k in range(30):

        state, _ = cube.reset()
        viewer.update_position()
        print("New challenge ...")
        time.sleep(1.0)

        for k in range(disorder_turns):
            action = agent.act(state)
            state, reward, done, _, _ = cube.step(action, with_faces = True)
            viewer.update_position()
            print(f"Iter {k} with reward: {reward}")
            time.sleep(0.5)
        

        time.sleep(1.0)
    
