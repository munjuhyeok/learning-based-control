import numpy as np
from environment import GridWorld
import random
import copy


class Agent:
    def __init__(self, environment, discount_factor=0.99, epsilon=0.2, learning_rate=0.01):
        self.env = environment
        self.actions = [0, 1, 2, 3]

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        ################

        # initialize Q, with -inf for impossible action
        self.q_table = np.zeros(env.size + [len(self.actions)])
        self.q_table[0, :, 0] = float("-inf")
        self.q_table[env.size[0] - 1, :, 1] = float("-inf")
        self.q_table[:, 0, 2] = float("-inf")
        self.q_table[:, env.size[1] - 1, 3] = float("-inf")

        self.policy = np.zeros(env.size, np.int)
        self.transition = []
        epsilon /= 0.99999

        #################

    def reset(self):
        # reset the agent each episode
        ################

        self.epsilon *= 0.99999
        self.transition = []

        #################

    def act(self, state):
        # sample an available action
        ################

        # your code

        #################
        return action

    def update_policy(self):
        # make greedy policy w.r.t. the value function
        ################

        # your code

        #################

    def store_transition(self, state, action, reward, next_state):
        # store the transition for offline update
        ################

        # your code

        #################

    def update(self):
        # update the value function
        ################

        # your code

        #################


if __name__ == '__main__':
    env = GridWorld()
    agent = Agent(environment=env)

    for episode in range(1000000):

        # uniformly sample the initial state. reject the terminal state
        ################

        while (True):
            init_state = []
            for i in env.size:
                init_state.append(random.randrange(i))
            if(init_state != env.goal):
                break
        #################

        state = env.reset(init_state)
        agent.reset()
        steps = 0

        # actual training
        while True:
            ################

            # your code

            #################
            if done or steps >= 1000:
                break
        agent.update()

        if episode % 10000 == 0:
            print("episode : {}, steps : {}".format(episode, steps))

    agent.update_policy()
    np.save("prob4_qlearning_policy.npy", agent.policy)
    np.save("prob4_qlearning_q_table.npy", agent.q_table)
