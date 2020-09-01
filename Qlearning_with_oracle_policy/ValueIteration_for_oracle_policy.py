
"""
Solving the simple cliff environment with value iteration to obtain qvalues at the "critical" states.
We treat them as the "oracle" values/policies, which are known beforehand when learning a policy for this environment.
Our hypothesis is that knowing the policies at the critical states should help in learning a better policy!
"""

import numpy as np
import pickle
import sys
from collections import defaultdict
from pprint import pprint

if "../" not in sys.path:
    sys.path.append("../")

from CliffEnv.lib.envs.cliff_walking import CliffWalkingEnv


def value_iteration(env, max_iterations=500000, lmbda=0.99):
  stateValue = [0 for i in range(env.nS)]
  newStateValue = stateValue.copy()
  for i in range(max_iterations):
    for state in range(env.nS):
      action_values = []
      for action in range(env.nA):
        state_value = 0
        for i in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][i]
          state_action_value = prob * (reward + lmbda*stateValue[next_state])
          state_value += state_action_value
        action_values.append(state_value)      #the value of each action
        best_action = np.argmax(np.asarray(action_values))   # choose the action which gives the maximum value
        newStateValue[state] = action_values[best_action]  #update the value of the state
    if i > 1000:
      if sum(stateValue) - sum(newStateValue) < 1e-04:   # if there is negligible difference break the loop
        print(i)
        break
    else:
      stateValue = newStateValue.copy()
  return stateValue


def get_q_values_and_policy(env,stateValue, lmbda=0.99):
  policy = [0 for i in range(env.nS)]
  Q = defaultdict(lambda: np.zeros(env.action_space.n))
  for state in range(env.nS):
    action_values = []
    for action in range(env.nA):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + lmbda * stateValue[next_state])
      action_values.append(action_value)
    Q[state] = np.asarray(action_values)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action
  return Q,policy


def save_policy_at_critical_states(Q):

  ## These critical states are the states neighboring the cliff - can be obtained by looking at their qvalues
  critical_states = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36]

  final_q_at_critical_states = {}
  for state in critical_states:
      final_q_at_critical_states[state] = Q[state]

  with open("final_qvalues_at_critical_states_with_valueiteration.pkl",'wb') as f:
      pickle.dump(final_q_at_critical_states,f)


if __name__ == "__main__":

    env = CliffWalkingEnv()
    env.render()

    stateValue = value_iteration(env)
    Q,policy = get_q_values_and_policy(env, stateValue)
    pprint(Q)

    save_policy_at_critical_states(Q)











