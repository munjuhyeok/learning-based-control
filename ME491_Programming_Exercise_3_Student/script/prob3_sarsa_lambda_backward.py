import numpy as np
from environment import GridWorld


class Agent:
    def __init__(self, environment, discount_factor=0.99, epsilon=0.2, lamb=0.8, learning_rate=0.01):
        self.env = environment
        self.actions = [0, 1, 2, 3]

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lamb = lamb
        self.learning_rate = learning_rate

        ################

        # your code

        #################

    def reset(self):
        # reset the agent each episode
        ################

        # your code

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

    def update(self, state, action, reward, next_state):
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

        # your code

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

        if episode % 10000 == 0:
            print("episode : {}, steps : {}".format(episode, steps))

    agent.update_policy()
    np.save("prob3_sarsa_backward_policy.npy", agent.policy)
    np.save("prob3_sarsa_backward_q_table.npy", agent.q_table)
