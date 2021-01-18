import gym
import numpy as np
from gym.spaces import Discrete
import matplotlib.pyplot as plt
import os

env = gym.make("Taxi-v3")

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
assert isinstance(env.action_space, Discrete)
assert isinstance(env.observation_space, Discrete)

class qlearning:

    # def __init__(self, env, name, training=True, gamma=0.99, Q=None ):
        """
                This Q-learning implementation only works on an environment with discrete
                action and observation space. We use a dict to memorize the Q-value.

                1. We start from state s and

                2.  At state s, with action a, we observe a reward r(s, a) and get into the
                next state s'. Update Q function:

                    Q(s, a) += learning_rate * (r(s, a) + gamma * max Q(s', .) - Q(s, a))

                Repeat this process.
                """
        # super().__init__(env,name, gamma=gamma, training=training)
        # assert isinstance(env.action_space, Discrete)
        # assert isinstance(env.observation_space, Discrete)
        #
        # self.Q = Q
        # self.actions = range(self.env.action_space.n)

        @staticmethod
        def build(env):
            #initializing qtable with zero values
            Qtable = np.zeros([env.observation_space.n, env.action_space.n])
            return Qtable

        @staticmethod
        def act(Q, training, state, eps=0.1):
            """Pick best action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy."""
            if training and eps > 0 and eps > np.random.rand():
                    return env.action_space.sample()

            else:
                # Pick the action with highest Q value.

                max_value_actions = np.argwhere(Q[state] == np.amax(Q[state]))
                # print(max_value_actions)
                return np.random.choice(max_value_actions.flatten())

        @staticmethod
        def update_q(Q, state_curr, state_next, action_curr, curr_r, gamma = 0.99,  alpha=0.01):
            """
            Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
            """
            curr_qvalue = Q[state_curr,action_curr]
            max_q_nextvalue = max(Q[state_next])

            Q[state_curr,action_curr] = (1-alpha)*curr_qvalue + alpha*(curr_r + (gamma * max_q_nextvalue))
            return Q

        @staticmethod
        def plot_learning_curve(value_dict, xlabel='step'):
            # Plot step vs the mean(last 50 episodes' rewards)
            fig = plt.figure(figsize=(12, 4 * len(value_dict)))

            for i, (key, values) in enumerate(value_dict.items()):
                ax = fig.add_subplot(len(value_dict), 1, i + 1)
                ax.plot(range(len(values))[-1000:], values[-1000:])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(key)
                ax.grid('k--', alpha=0.6)

            plt.tight_layout()

            plt.savefig('plot' + str(len(values)) + ".png")






