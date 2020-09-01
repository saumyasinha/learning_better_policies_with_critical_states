import pickle as pkl
from matplotlib import pyplot as plt
import pandas as pd


with open('Stats/stats_qlearning_with_oracle_policy_for_cliffenv.pickle', 'rb') as f:
    stats_upper_bound = pkl.load(f)

with open('Stats/stats_vanillaq_for_cliffenv.pickle', 'rb') as f:
    stats_vanilla = pkl.load(f)


## overlay the two reward plots for a better comparison
smoothing_window=10
fig2 = plt.figure(figsize=(10,5))

stats_vanilla_smoothed = pd.Series(stats_vanilla).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(stats_vanilla_smoothed, color = 'y', label = 'vanilla_Qlearning')

upper_bound_rewards_smoothed = pd.Series(stats_upper_bound).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(upper_bound_rewards_smoothed, color = 'b', label = 'Qlearning_with_oracle_policy')

plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.legend(loc="lower right")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
plt.savefig('plots/overlaid_plots_of_vanillaQlearning_vs_Qlearning_with_oracle.png')
