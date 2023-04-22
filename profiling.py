from datetime import datetime, timedelta


import gym
import nle
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


from agents.torchbeast_agent import TorchBeastAgent
from agents.custom_agent import CustomAgent
from create_graph_representation import TileGraphV1, TileGraphV2

env = gym.make("NetHackScore-v0")
agent = CustomAgent(1, 23) # 1 env, 23 possible actions
# agent = TorchBeastAgent(1, 23)
graph1 = TileGraphV1()
graph2 = TileGraphV2()

total_times = []
gnn_times1 = []
gnn_times2 = []

for trial in tqdm(range(12)):
    out = env.reset()
    out = env.step(action=0)

    total_time = 0.0
    GNN_time1 = timedelta(0)
    GNN_time2 = timedelta(0)

    start = datetime.now()
    for step in range(4000):
        (obs, rewards, done, infos) = out
        action = agent.batched_step(obs, rewards, done, infos)
        start2 = datetime.now()
        graph1.update(obs=obs)
        GNN_time1 += datetime.now() - start2
        start2 = datetime.now()
        graph2.update(obs=obs)
        GNN_time2 += datetime.now() - start2

        try:
            env.step(action=action.item())
        except RuntimeError:
            break
            # Sometimes env is finished but done flag isn't set?....

    total_time = datetime.now() - start
    total_times.append((total_time-GNN_time2-GNN_time1).microseconds)
    gnn_times1.append(GNN_time1.microseconds)
    gnn_times2.append(GNN_time2.microseconds)

gnn_times1 = np.array(gnn_times1) / 1000
gnn_times2 = np.array(gnn_times2) / 1000
total_times = np.array(total_times) / 1000

print(f"Graph Updates V1: {np.mean(gnn_times1):.2f} s +/- {np.std(gnn_times1):.2f}")
print(f"Graph Updates V2: {np.mean(gnn_times2):.2f} s +/- {np.std(gnn_times2):.2f}")
print(f"Simulation: {np.mean(total_times):.2f} s +/- {np.std(total_times):.2f}")

fig, axes = plt.subplots(1, 3)

axes[0].boxplot(total_times)
axes[1].boxplot(gnn_times1)
axes[2].boxplot(gnn_times2)

ymin = -5
ymax = max(max(total_times), max(gnn_times1))+5

plt.setp(axes, ylim=(ymin, ymax))


axes[0].set_title('Simulation Runtime (s)')
axes[1].set_title('Graph Update Runtime V1 (s)')
axes[2].set_title('Graph Update Runtime V2 (s)')

fig.tight_layout()
fig.savefig('./fig.png')
