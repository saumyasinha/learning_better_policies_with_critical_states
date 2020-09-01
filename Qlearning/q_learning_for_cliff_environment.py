
import random
import time
import itertools
import matplotlib
import numpy as np
import sys
import pickle
from pprint import pprint


if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from CliffEnv.lib.envs.cliff_walking import CliffWalkingEnv
from CliffEnv.lib import plotting

matplotlib.style.use('ggplot')

# UP = 0
# RIGHT = 1
# DOWN = 2
# LEFT = 3

def epsilon_greedy(Q, epsilon, state):
    random_number = random.random()
    if random_number < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])



def q_learning(env, num_episodes, discount_factor=0.99, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: (OpenAI) environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    all_rewards = []
    start = time.time()

    for i_episode in range(num_episodes):

        # Reset the environment
        state = env.reset()

        for t in itertools.count():

            # Pick the action using epsilon-greedy strategy
            action = epsilon_greedy(Q, epsilon, state)
            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                all_rewards.append(stats.episode_rewards[i_episode])
                avg_reward = float(np.mean(all_rewards[-100:]))
                # print('Best 100-episodes average reward', i_episode, avg_reward)
                break

            state = next_state

    print("training time", time.time()-start)
    return Q, stats



def play(Q):

    initial_state_list = [25, 29, 30, 33, 36]
    avg_test_reward = []
    for t in range(5):
        print('play: ',t)
        state = initial_state_list[t]
        # print(state)
        env.custom_reset(state)
        env.render()
        done = False
        reward_per_episode = 0
        while not done:
            action, explore = epsilon_greedy(Q, 0, state)
            print(action)
            next_state, reward, done, info = env.step(action)
            env.render()
            reward_per_episode+=reward

            if done:
                print('rewards: ',reward_per_episode)
                avg_test_reward.append(reward_per_episode)
                break
            else:
                state = next_state

    print(np.mean(avg_test_reward))


# def save_policy_at_critical_states(Q):
#
#     critical_states = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36]
#
#     final_q_at_critical_states = {}
#     for state in critical_states:
#         final_q_at_critical_states[state] = Q[state]
#
#     with open("final_q_at_critical_states.pkl",'wb') as f:
#         pickle.dump(final_q_at_critical_states,f)


def find_critical_states(thresh, Q):
    """
    Using the critical state definition from Sandy Huang's work:
    "Establishing Appropriate Trust via Critical States"
    """
    critical_states = []

    for state in range(len(Q)):

        maxq = np.max(Q[state])
        meanq = np.mean(Q[state])

        if maxq - meanq > thresh:
            critical_states.append(state)
            # env.render()

    print(critical_states)


if __name__ == "__main__":

    env = CliffWalkingEnv()
    env.render()

    num_episodes = 100000
    Q, stats = q_learning(env, num_episodes)

    pprint(Q)

    f = open('stats_vanilla.pickle', 'wb')
    pickle.dump(stats.episode_rewards, f)

    plotting.plot_episode_stats(stats, "./plots/vanilla_q_learning.png", noshow = True)






