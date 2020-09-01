import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from collections import deque

env_id = "CartPole-v0"
env = gym.make(env_id)
# env = env.unwrapped

path = "/Users/saumya/Desktop/CriticalStates_results/"
results_dir = "vanillaDQN"

'''
Double DQN code adapted and modified from https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
'''

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            # nn.Linear(env.observation_space.shape[0], 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, env.action_space.n)

            # Function approximator for Q function - modified to less hidden neurons
            nn.Linear(env.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        """
        choose action using epsilon-greedy strategy
        """
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(batch_size):
    """
    Compute the TD loss after sampling transitions(of size - "batch_size") from the replay buffer
    """
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot(frame_idx, rewards, losses, iter):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    # plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.title('frame %s' % (frame_idx))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(path+results_dir+"/cartpole_dqn_plots_iter_"+str(iter))


def load_model(model_path):
    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    current_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    return current_model


def play(model_path):
    """
    Play or rollout the learnt policy and observe the mean reward obtained over 1000 episodes
    """
    current_model = load_model(model_path)
    avg_test_reward = []
    for t in range(1000):
        # print('play: ',t)
        state = env.reset()
        done = False
        reward_per_episode = 0
        while not done:

            action = current_model.act(state, 0)
            next_state, reward, done, info = env.step(action)
            # env.render()
            reward_per_episode+=reward

            if done:
                # print('rewards: ',reward_per_episode)
                avg_test_reward.append(reward_per_episode)
                break
            else:
                state = next_state

    env.close()

    print(np.mean(avg_test_reward))



if __name__ == "__main__":
    ## Hyperparameters
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    num_frames = 400000 # increased num of timesteps from 160000
    batch_size = 64
    gamma = 0.99
    update_target_net = 100
    learning_rate = 1e-4 # reduced learning rate from 1e-3

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    ## Running for 5 iteration to obtain a mean and std of the reward plots
    for iter in range(5):

        print("iteration: ",iter)

        current_model = DQN(env.observation_space.shape[0], env.action_space.n)
        target_model = DQN(env.observation_space.shape[0], env.action_space.n)


        if USE_CUDA:
            current_model = current_model.cuda()
            target_model = target_model.cuda()

        optimizer = optim.Adam(current_model.parameters(), lr = learning_rate)

        replay_buffer = ReplayBuffer(100000) # increased buffer size from 1000
        update_target(current_model, target_model)

        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0

        ## If the environment is solved is_win is set true
        is_win = False
        state = env.reset()
        for frame_idx in range(1, num_frames + 1):

            epsilon = epsilon_by_frame(frame_idx)
            action = current_model.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num+=1
                avg_reward = float(np.mean(all_rewards[-100:]))
                print('Best 100-episodes average reward', ep_num, avg_reward)

                ## Using the following "solving" criteria
                if len(all_rewards) >= 100 and avg_reward >= 198 and all_rewards[-1] > 198:
                    if not is_win:
                        is_win = True
                        torch.save(current_model.state_dict(), path+results_dir+'/CartPole_dqn_model_iter_'+str(iter))
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (
                    ep_num, avg_reward, ep_num - 100))
                    last_saved = ep_num
                    torch.save(current_model.state_dict(),
                                path+results_dir+'/Final_CartPole_dqn_model_iter_' + str(
                                   iter))

            ## Update the loss
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.item())

            if frame_idx % 200 == 0:
                plot(frame_idx, all_rewards, losses, iter)

            ## Update the target network
            if frame_idx % update_target_net == 0:
                update_target(current_model, target_model)

        ## Save the reward list - rewards obtained per episode
        np.save(path+results_dir+"/rewards_iter_"+str(iter),all_rewards)

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            torch.save(current_model.state_dict(), path+results_dir+'/CartPole_dqn_model_iter_'+str(iter))


        # play(path+results_dir+'/CartPole_dqn_model_iter_'+str(iter))
        # play(path+results_dir+'/Final_CartPole_dqn_model_iter_' + str(iter))
        # Iteration: 0
        # 199.969
        # 200.0
        # iteration: 1
        # 200.0
        # 195.842
        # iteration: 2
        # 200.0
        # 182.442
        # iteration: 3
        # 200.0
        # 200.0
        # iteration: 4
        # 197.461
        # 199.972
