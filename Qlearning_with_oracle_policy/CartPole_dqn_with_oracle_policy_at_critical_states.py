import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Qlearning.dqn_for_CartPole import DQN as vanillaDQN

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from collections import deque

'''
Double DQN code adapted and modified from https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
'''

env_id = "CartPole-v0"
env = gym.make(env_id)

path = "/Users/saumya/Desktop/CriticalStates_results/"
results_dir = "DQN_with_oracle_policy_at_critical_states"

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.critical_buffer = []

    def push(self, state, action, reward, next_state, done, is_critical):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done, is_critical))

    def sample(self, batch_size):
        state, action, reward, next_state, done, is_critical = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done, is_critical

    def __len__(self):
        return len(self.buffer)+len(self.critical_buffer)



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        ## Function approximator for Q function
        self.layers = nn.Sequential(
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
        Choose action with epsilon-greedy strategy except we now don't explore at the critical states
        """
        if (random.random() < epsilon) and (is_critical(state)==False):
            action = random.randrange(env.action_space.n)
        else:
            tensor_state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            if is_critical(state):
                q_value = oracle(tensor_state)
            else:
                q_value = self.forward(tensor_state)
            action = q_value.max(1)[1].item()

        return action


def is_critical(state, thresh=5):
    '''
    :param state: state as a numpy array
    :param thresh: threshold to classify a state as a critical state
    :return: true if the state is critical according to the definition in Sandy Huang's work:
    "Establishing Appropriate Trust via Critical States", else false
    '''
    state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
    q_value = oracle(state)
    advantage = torch.max(q_value).item() - torch.mean(q_value).item()
    if advantage > thresh:
        # print(q_value)
        return True

    else:
        return False


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_loss(batch_size, critical_loss_weight=0.1):
    """
    :param batch_size: number of transitons(s,a,s',r) to be sampled from the replay buffer for loss calculation
    :param critical_loss_weight: a number < 1, to lower the weight of loss calculated for critical states
    :return: loss values
    """
    state, action, reward, next_state, done, is_critical = replay_buffer.sample(batch_size)

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

    ## is_critical is a boolean tensor keeping track of the states that are classified as critical
    is_critical = torch.Tensor(is_critical) == True

    ## TD_loss is the usual loss calculated in vanilla DQN, this loss is calculated only for states that are not critical
    TD_loss = (q_value[is_critical==0] - Variable(expected_q_value[is_critical==0].data)).pow(2).mean()

    critical_loss = torch.zeros(1)
    if is_critical.sum()>0:
        ## If atleast a single state is critical:
        # we define the critical_loss for them as the difference between current q values and the oracle's q values
        expected_q_value[is_critical] = oracle(state[is_critical]).gather(1, action[is_critical].unsqueeze(1)).squeeze(1)
        critical_loss = (q_value[is_critical] - Variable(expected_q_value[is_critical].data)).pow(2).mean()

    ## Taking a weighted sum of the TD and critical loss, as the TD loss is expected to be much lower
    loss = TD_loss + critical_loss_weight*critical_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return TD_loss, critical_loss


def plot(frame_idx, rewards, TD_losses, critical_losses, iter):
    # clear_output(True)
    plt.figure(figsize=(30,5))
    plt.subplot(131)
    plt.title('frame %s' % (frame_idx))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('critical loss')
    plt.plot(critical_losses)
    plt.subplot(133)
    plt.title('TD loss')
    plt.plot(TD_losses)
    plt.savefig(path+results_dir+"/CartPole_dqn_plots_using_weighted_loss_"+str(iter))


def load_model(model_path):
    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    current_model.load_state_dict(torch.load(model_path))

    return current_model

def play(model_path):
    """
    Play or rollout the learnt policy and observe the mean reward obtained over 1000 episodes
    """
    current_model = load_model(model_path)
    avg_test_reward = []
    for t in range(1000):
        print('play: ',t)
        state = env.reset()
        env.render()
        done = False
        reward_per_episode = 0
        while not done:

            action = current_model.act(state, 0)
            next_state, reward, done, info = env.step(action)
            env.render()
            reward_per_episode+=reward

            if done:
                print('rewards: ',reward_per_episode)
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
    num_frames = 400000
    batch_size = 64
    gamma = 0.99
    update_target_net = 100
    learning_rate = 1e-4

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    ## Load the "Oracle" model to obtain q values of the critical states (Oracle is a fully converged double DQN model)
    oracle = vanillaDQN(env.observation_space.shape[0], env.action_space.n)
    oracle.load_state_dict(
        torch.load(path + 'vanillaDQN/Final_CartPole_dqn_model_with_increased_frames_and_lower_lr_iter_3',
                   map_location=torch.device('cpu')))

    ## Running for 5 iteration to observe the variance in learning
    for iter in range(5):

        print("iteration: ", iter)

        current_model = DQN(env.observation_space.shape[0], env.action_space.n)
        target_model = DQN(env.observation_space.shape[0], env.action_space.n)


        if USE_CUDA:
            current_model = current_model.cuda()
            target_model = target_model.cuda()

        optimizer = optim.Adam(current_model.parameters(), lr = learning_rate)

        replay_buffer = ReplayBuffer(100000)
        update_target(current_model, target_model)

        critical_losses = []
        TD_losses = []
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0

        ## If the environment is solved is_win is set to true
        is_win = False

        state = env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_by_frame(frame_idx)
            action = current_model.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            ## Add an additional boolean on whether the state is critical or not (to be used when calculating loss)
            replay_buffer.push(state, action, reward, next_state, done, is_critical(state))

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
                    is_win = True
                    torch.save(current_model.state_dict(), path+results_dir+'/CartPole_dqn_model_using_weighted_loss_iter_'+str(iter))
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (
                    ep_num, avg_reward, ep_num - 100))


            ## Update the loss
            if len(replay_buffer) > batch_size:
                TD_loss, critical_loss = compute_loss(batch_size)
                TD_losses.append(TD_loss.item())
                critical_losses.append(critical_loss.item())

            if frame_idx % 200 == 0:
                plot(frame_idx, all_rewards, TD_losses, critical_losses, iter)

            ## Update the target network
            if frame_idx % update_target_net == 0:
                update_target(current_model, target_model)

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            torch.save(current_model.state_dict(), path+results_dir+'/CartPole_dqn_model_using_weighted_loss_iter_'+str(iter))

        ## Save the reward list - rewards obtained per episode
        np.save(path+results_dir+"/rewards_using_weighted_loss_iter_" + str(iter), all_rewards)

