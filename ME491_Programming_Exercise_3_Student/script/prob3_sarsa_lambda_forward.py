import numpy as np
from environment import GridWorld
import random
import copy


class Agent:
    def __init__(self, environment, discount_factor=0.99, epsilon=0.2, lamb=0.8, learning_rate=0.01):
        self.env = environment
        self.actions = [0, 1, 2, 3]

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lamb = lamb
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

        if random.random() < self.epsilon:
            action = random.choice(env.get_possible_actions(state))
        else:
            action = np.argmax(self.q_table[tuple(state)])

        #################
        return action

    def update_policy(self):
        # make greedy policy w.r.t. the value function
        ################

        self.policy = np.argmax(self.q_table, axis=-1)

        #################

    def store_transition(self, state, action, reward, next_state):
        # store the transition for offline update
        ################

        self.transition.append([state, action, reward, next_state])

        #################

    def update(self):
        # update the value function
        ################

        transition_length = len(self.transition)

        returns = np.asarray([])
        coefficient = np.array([1])
        next_action = self.act(self.transition[-1][-1])
        q_table_temp = self.q_table.copy()

        for step in reversed(range(transition_length)):
            [state, action, reward, next_state] = self.transition[step]
            if next_state == env.goal:
                returns = np.array([reward])
            else:
                returns = reward + self.discount_factor * np.concatenate(([q_table_temp[tuple(next_state)][next_action]],returns))
            next_action = action
            self.q_table[tuple(state)][action] += self.learning_rate * coefficient.dot(returns-self.q_table[tuple(state)][action])
            coefficient = np.concatenate(([(1 - self.lamb)], self.lamb*coefficient))

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

        state = env.reset(copy.deepcopy(init_state))
        agent.reset()
        steps = 0

        # actual training
        while True:
            ################

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            steps += 1

            #################
            if done or steps >= 1000:
                break
        agent.update()

        if episode % 10000 == 0:
            print("episode : {}, steps : {}".format(episode, steps))

    agent.update_policy()
    np.save("prob3_sarsa_forward_policy.npy", agent.policy)
    np.save("prob3_sarsa_forward_q_table.npy", agent.q_table)
