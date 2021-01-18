import numpy as np
import QLearning
from QLearning import qlearning

class TrainMyRLAgent:
    def __init__(self, env, gamma , training, alpha, alpha_decay, epsilon, epsilon_final, n_episodes,
                 warmup_episodes, log_every_episode ):

        Qtable = qlearning.build(env)

        reward_history=[]
        averaged_reward = []
        all_epochs = []
        all_penalties = []


        warmup_episodes = warmup_episodes or n_episodes
        eps_drop = (epsilon - epsilon_final) / warmup_episodes

        print("num of episodes:",n_episodes)

        for episode in range(1,n_episodes):

            state = env.reset()
            done = False
            epochs, penalties, reward, = 0, 0, 0.

            while not done:
                curr_action = qlearning.act(Qtable ,training, state, eps=epsilon)
                next_state, r, done, info = env.step(curr_action)
                # env.step(action): Step the environment by one timestep. Returns
                # observation: Observations of the environment
                # reward: If your action was beneficial or not
                # done: Indicates if we have successfully picked up and dropped off a passenger, also called one episode
                # info: Additional info such as performance and latency for debugging purposes

                # if done and config.done_reward is not None:
                #     r += config.done_reward

                Qtable = qlearning.update_q(Qtable,state,next_state,curr_action,r,gamma, alpha)
                if r == -10:
                    penalties+=1

                epochs+=1
                state = next_state
                reward+=r

            reward_history.append(reward)
            averaged_reward.append(np.average(reward_history[-50:]))

            all_penalties.append(penalties)
            all_epochs.append(epochs)



            if epsilon > epsilon_final:
                epsilon = max(epsilon_final, epsilon - eps_drop)

            if (log_every_episode is not None) and (episode % log_every_episode == 0):
                # print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} currreward:{}".format(
                #     episode, epochs, np.max(reward_history),
                #     np.mean(reward_history[-10:]), alpha, epsilon, reward))
                alpha *= alpha_decay

        print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} penalties_avg:{}".format(
                episode, epochs, np.max(reward_history),
                np.mean(reward_history[-100:]), alpha, epsilon, np.mean(all_penalties[-100:]) ))
        print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))
        # print(all_penalties[50:])
        # print(all_penalties[-10:])
        data_dict = {'reward': reward_history, 'reward_avg50': averaged_reward}
        qlearning.plot_learning_curve(data_dict, xlabel='episode')











