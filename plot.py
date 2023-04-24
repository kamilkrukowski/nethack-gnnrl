import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


save_dir = '/Users/kamilkrukowski/torchbeast/'
options = os.listdir(save_dir)

# choice = save_dir + 'torchbeast-20230422-164556/'
choice = save_dir + 'latest/'

def get_results(dir_path: str):
    return pd.read_csv(dir_path + '/logs.tsv', sep='\t')

df = get_results(choice)

res = df[['# Step', 'mean_episode_return', 'total_loss', 'pg_loss',
          'baseline_loss']].dropna()

# Smoothing of median return window size
K = 1000
rolling = np.lib.stride_tricks.sliding_window_view(res.mean_episode_return, K)
smooth = np.mean(rolling, axis=1)
stderrs = np.std(rolling, axis=1)

# Smoothing of confidence intervals
K2 = 1000
stderr_upper = smooth + stderrs
stderr_upper = np.percentile(rolling, 75, axis=1)
stderr_lower = smooth - stderrs
stderr_lower = np.percentile(rolling, 25, axis=1)
ker = np.ones(K2)/float(K2)
stderr_upper = np.convolve(stderr_upper, ker, 'same')
stderr_lower = np.convolve(stderr_lower, ker, 'same')

# Set up X-axis units
xs = res['# Step'].values
xs = xs[0] + xs[-(len(smooth)):]

# Truncate to avoid smoothing boundary conditions for intervals.
start = int(K2/2.0)
end = int(K2/2.0)
smooth = smooth[start:-end]
xs = xs[start:-end]
stderr_upper = stderr_upper[start:-end]
stderr_lower = stderr_lower[start:-end]

plt.plot(xs, smooth, label='Median Return')
plt.fill_between(xs, stderr_lower, stderr_upper, alpha=0.2, label='+/- 1 STD')

plt.title('Episode Returns')
#plt.legend(loc='best').set_draggable(True)
plt.legend(loc='lower right').set_draggable(True)

plt.xlabel('Simulation Steps')
plt.ylabel("Score")

plt.tight_layout()
plt.savefig('./latest_returns.png')
