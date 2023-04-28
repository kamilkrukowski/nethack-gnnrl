import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_results(dir_path: str):
    return pd.read_csv(dir_path + '/logs.tsv', sep='\t')


def custom_plot(res, K=3, K2=3, label='Median Return'):
    # Smoothing of median return window size
    rolling = np.lib.stride_tricks.sliding_window_view(res.mean_episode_return, K)
    smooth = np.mean(rolling, axis=1)
    stderrs = np.std(rolling, axis=1)

    # Smoothing of confidence intervals
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
    start = 0
    end = 1
    smooth = smooth[start:-end]
    xs = xs[start:-end]
    stderr_upper = stderr_upper[start:-end]
    stderr_lower = stderr_lower[start:-end]

    plt.plot(xs, smooth, label=label)
    plt.fill_between(xs, stderr_lower, stderr_upper, alpha=0.2)

    plt.title('Episode Returns')
    #plt.legend(loc='best').set_draggable(True)
    plt.legend(loc='best').set_draggable(True)

    plt.xlabel('Simulation Steps')
    plt.ylabel("Score")

    plt.tight_layout()

save_dir = '/Users/kamilkrukowski/torchbeast/'
options = os.listdir(save_dir)

# choice = save_dir + 'torchbeast-20230422-164556/'
choice = save_dir + 'latest/'
todo = []
df1 = get_results(save_dir + 'baselinegnn')
df2 = get_results(save_dir + 'posenc')
df3 = get_results(choice)
df4 = get_results(save_dir + 'base')

paths = [save_dir + 'baselinegnn', save_dir + 'posenc', save_dir + 'base', save_dir + 'latest/']
labels = ['BaseGNN', 'GAT + POSENC', 'BaseCNN', 'Graph-RNN']
res = []
for path in paths:
    df = get_results(path)
    res.append(df[['# Step', 'mean_episode_return', 'total_loss', 'pg_loss',
                    'baseline_loss']].dropna())

clip = False
if clip:
    minL = min([max(i['# Step'] for i in res)])
    for i, label in zip(range(len(res)), labels):
        res[i] = res[i][res[i]['# Step'] < minL]

K  = 500
K2 = 1000
P = 0.3
K = min(max(min([int(len(i)*P) for i in res]), 2), K)
K2 = K

for i, label in zip(range(len(res)), labels):
    custom_plot(res[i], K=K, K2=K2, label=label)

res1 = res[0]
res2 = res[1]
res3 = res[2]
xmin = max([i['# Step'].iloc[K2+K] for i in res])
xmax = min([i['# Step'].iloc[K2+K] for i in res])
xmax = 1.4e+7
plt.xlim((xmin, xmax))
plt.tight_layout()
plt.savefig('./latest_returns.png')
