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
    """
    Usual epsilon greedy strategy except that we no longer explore when at the critical states
    """
    random_number = random.random()
    if (random_number < epsilon) and (state not in critical_states):
        return env.action_space.sample()

    else:
        return np.argmax(Q[state])


def q_learning(env, num_episodes,final_q_at_critical_states, discount_factor=0.99, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        final_q_at_critical_states: q values at critical states which are obtained from an oracle and known beforehand
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

    # Update the Q table with the known q values at the critical states
    for state in critical_states:
        Q[state] = final_q_at_critical_states[state]

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

            # TD Update, but only for states that are not critical
            if state not in critical_states:
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

            if done:
                all_rewards.append(stats.episode_rewards[i_episode])
                # print(i_episode,stats.episode_rewards[i_episode])
                break

            state = next_state

    print("training time", time.time()-start)
    return Q, stats


def load_policy_at_critical_states():

    with open("final_qvalues_at_critical_states_with_valueiteration.pkl",'rb') as f:
        return pickle.load(f)


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

            action = epsilon_greedy(Q, 0, state)
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



if __name__ == "__main__":
    env = CliffWalkingEnv()
    env.render()

    num_episodes = 100000

    ## critical states are the states obtained from the definition in this paper --> "Establishing Appropriate Trust via Critical States"
    critical_states = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36]

    final_q_at_critical_states = load_policy_at_critical_states()
    Q, stats = q_learning(env, num_episodes, final_q_at_critical_states)
    pprint(Q)

    f = open('Stats/stats_qlearning_with_oracle_policy_for_cliffenv.pickle', 'wb')
    pickle.dump(stats.episode_rewards, f)

    plotting.plot_episode_stats(stats, "./plots/CliffWalking/q_learning_with_oracle_policy_at_critical_states.png", noshow = True)








