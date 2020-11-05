import numpy as np
import torch
from collections import deque
from dql_agent import Agent
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


class Arena:
    """Arena controls training executions"""
    def __init__(self, global_env):
        self.env=global_env
        # Environments contain brains which are responsible for deciding the actions of their associated agents
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.agent = Agent(state_size=37, action_size=4, seed=0)
        self.threshold = 15.0

    def train_dqn(self, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[self.brain_name] # reset the environment
            state = env_info.vector_observations[0]            # get the current state
            score = 0                                          # initialize the score
            for t in range(max_t):
                action = self.agent.act(state, eps)
                env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]                  # see if episode has finished
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            self.scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            # we use 15.0 just to be sure
            if np.mean(scores_window)>=self.threshold:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break
        return self.scores

    def save_model(self, filename):
        torch.save(self.agent.qnetwork_local.state_dict(), filename)

    def plot_scores(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        