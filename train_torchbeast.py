# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is an example self-contained agent running NLE based on MonoBeast.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
from tqdm.auto import tqdm

from torch_geometric.utils import coalesce
from torch.nn.functional import pad
from numpy import nan

from create_graph_representation import TileGraphPosEnc as TileGraph
from nethacknet import NetHackNet

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
    from torch import multiprocessing as mp
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    logging.exception(
        "PyTorch not found. Please install the agent dependencies with "
        '`pip install "nle[agent]"`'
    )

import gym  # noqa: E402

import nle  # noqa: F401, E402
from nle import nethack  # noqa: E402
from nle.agent import vtrace  # noqa: E402

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, default="NetHackScore-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/torchbeast/",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=1, type=int, metavar="N",
                    help="Number of actors (default: 1).")
parser.add_argument("--total_steps", default=20000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--pyg_nodes_max", default=800, type=int, metavar="T",
                    help="Maximum number of PYGDATA nodes")
parser.add_argument("--pyg_node_fdim", default=4, type=int, metavar="T",
                    help="PYGDATA Node feature dimension")
parser.add_argument("--pyg_edges_max", default=4000, type=int, metavar="T",
                    help="Maximum number of PYGDATA edges")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument("--resume", action="store_true",
                    help="Resume training")
parser.add_argument("--resume_path", default="latest",
                    help="Directory to resume training from")
parser.add_argument("--save_ttyrec_every", default=1000, type=int,
                    metavar="N", help="Save ttyrec every N episodes.")


# Loss settings.
parser.add_argument("--entropy_cost", default=0.009,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=logging.INFO,
)


def nested_map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(nested_map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: nested_map(f, v) for k, v in n.items()}
    else:
        return f(n)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages**2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def create_env(name, *args, **kwargs):
    return gym.make(name, observation_keys=("glyphs", "blstats"), *args, **kwargs)


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)

        graph = TileGraph(max_nodes=flags.pyg_nodes_max,
                          max_edges=flags.pyg_edges_max)
        graph.reset()

        gym_env = create_env(
            flags.env, savedir=flags.rundir, save_ttyrec_every=flags.save_ttyrec_every
        )
        env = ResettingEnvironment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        graph.update(env_output)
        pygData = graph.pyg()
        env_output['pygData'] = pygData

        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                if key == 'pygData':
                    continue
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                env_output = env.step(agent_output["action"])

                if env_output['episode_step'] == 1:
                    graph.reset()

                graph.update(env_output)
                pygData = graph.pyg()
                env_output['pygData'] = pygData

                for key in env_output:
                    if key == 'pygData':
                        continue
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                buffers['pyg_n_nodes'][index][t + 1, ...] = len(pygData.x)
                buffers['pyg_nodes'][index][t + 1, :, :] = pad(
                    pygData.x, (0, 0, 0, flags.pyg_nodes_max - len(pygData.x)),
                    value=0)

                edge_index = coalesce(pygData.edge_index)
                buffers['pyg_n_edges'][index][t + 1, ...] = edge_index.shape[1]
                edge_index = pad(
                    edge_index, (
                        0, flags.pyg_edges_max - edge_index.shape[1], 0, 0),
                    value=0)
                buffers['pyg_edges'][index][t + 1, ...] = edge_index

            full_queue.put(index)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers,
    initial_agent_state_buffers,
    lock=threading.Lock(),
):
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    for m in indices:
        free_queue.put(m)
    batch = {k: t.to(device=flags.device, non_blocking=True)
             for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1]
                           for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, observation_space, num_actions, num_overlapping_steps=1):
    size = (flags.unroll_length + num_overlapping_steps,)

    # Get specimens to infer shapes and dtypes.
    samples = {k: torch.from_numpy(v)
               for k, v in observation_space.sample().items()}

    specs = {
        key: dict(size=size + sample.shape, dtype=sample.dtype)
        for key, sample in samples.items()
    }
    specs.update(
        reward=dict(size=size, dtype=torch.float32),
        done=dict(size=size, dtype=torch.bool),
        episode_return=dict(size=size, dtype=torch.float32),
        episode_step=dict(size=size, dtype=torch.int32),
        policy_logits=dict(size=size + (num_actions,), dtype=torch.float32),
        baseline=dict(size=size, dtype=torch.float32),
        last_action=dict(size=size, dtype=torch.int64),
        action=dict(size=size, dtype=torch.int64),
        pyg_n_nodes=dict(size=size, dtype=torch.int32),
        pyg_n_edges=dict(size=size, dtype=torch.int32),
        pyg_nodes=dict(size=(flags.unroll_length + num_overlapping_steps,
                       flags.pyg_nodes_max, flags.pyg_node_fdim), dtype=torch.int32),  # int32 is necessary for torch.nn.Embedding Layer
        pyg_edges=dict(size=(flags.unroll_length + num_overlapping_steps, 2,
                       flags.pyg_edges_max), dtype=torch.int64),
    )
    buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
#        buffers['pyg_edges'][-1] = buffers['pyg_edges'][-1]
    return buffers


def _format_observations(observation, keys=("glyphs", "blstats")):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations


class ResettingEnvironment:
    """Turns a Gym environment into something that can be step()ed indefinitely."""

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        result = _format_observations(self.gym_env.reset())
        result.update(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )
        return result

    def step(self, action):
        observation, reward, done, unused_info = self.gym_env.step(
            action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            observation = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        result = _format_observations(observation)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        result.update(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )
        return result

    def close(self):
        self.gym_env.close()


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    flags.savedir = os.path.expandvars(os.path.expanduser(flags.savedir))

    rundir = os.path.join(
        flags.savedir, "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    )
    if flags.resume:
        rundir = os.path.join(
            flags.savedir, flags.resume_path
        )

    if not os.path.exists(rundir):
        os.makedirs(rundir)
    logging.info("Logging results to %s", rundir)

    symlink = os.path.join(flags.savedir, "latest")
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(rundir, symlink)
        logging.info("Symlinked log directory: %s", symlink)
    except OSError:
        raise
    
    step, stats = 0, {}

    if flags.resume:
        step = open(os.path.join(rundir, "logs.tsv"), "r", buffering=1).read()
        step = int(step.strip().split('\n')[-1].split('\t')[0])
        print(f"RESUMING ON STEP {step}")

    logfile = open(os.path.join(rundir, "logs.tsv"), "a", buffering=1)
    checkpointpath = os.path.join(rundir, "model.tar")

    flags.rundir = rundir

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags.env)
    observation_space = env.observation_space
    action_space = env.action_space
    del env  # End this before forking.

    model = Net(observation_space, action_space.n, flags.use_lstm)
    
    if flags.resume:
        print(f"RESUMING FROM {flags.resume_path}")
        checkpointpath = os.path.join(flags.savedir, flags.resume_path, "model.tar")
        checkpoint = torch.load(checkpointpath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    buffers = create_buffers(flags, observation_space, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
#    manager = mp.Manager()
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
            name="Actor-%i" % i,
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(observation_space, action_space.n, flags.use_lstm).to(
        device=flags.device
    )
    learner_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]


    
    if not flags.resume:
        logfile.write("# Step\t%s\n" % "\t".join(stat_keys))
    else:
        logfile.write('\n')

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        while step < flags.total_steps:
            batch, agent_state = get_batch(
                flags, free_queue, full_queue, buffers, initial_agent_state_buffers
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            with lock:
                logfile.write("%i\t" % step)
                logfile.write("\t".join(str(stats[k]) for k in stat_keys))
                logfile.write("\n")
                step += T * B

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn,
            name="batch-and-learn-%d" % i,
            args=(i,),
            daemon=True,  # To support KeyboardInterrupt below.
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    pbar = tqdm(leave=True)
    mer = 0
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            # if stats.get("episode_returns", None):
            #    mean_return = (
            #        "Return per episode: %.1f. " % stats["mean_episode_return"]
            #    )
            # else:
            #    mean_return = ""
            # total_loss = stats.get("total_loss", float("inf"))
            if stats.get("episode_returns", None):
                mer = stats.get('mean_episode_return', mer)
            extra = f"latest_episode_return {mer:.2f}: "
            pbar.set_description(f"Steps {step:.2e} @ {sps:.1f} SPS: " + extra)
            # logging.info(
            #    "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
            #    step,
            #    sps,
            #    total_loss,
            #    mean_return,
            #    pprint.pformat(stats),
            # )
    except KeyboardInterrupt:
        logging.warning("Quitting.")
        pbar.close()
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    pbar.close()
    checkpoint()
    logfile.close()


def test(flags, num_episodes=10):
    flags.savedir = os.path.expandvars(os.path.expanduser(flags.savedir))
    checkpointpath = os.path.join(flags.savedir, "latest", "model.tar")

    gym_env = create_env(flags.env, save_ttyrec_every=flags.save_ttyrec_every)
    env = ResettingEnvironment(gym_env)
    model = Net(gym_env.observation_space,
                gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    graph = TileGraph(
        max_nodes=flags.pyg_nodes_max, max_edges=flags.pyg_edges_max)
    returns = []

    agent_state = model.initial_state(batch_size=1)

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()

        graph.update(observation)
        pygData = graph.pyg()
        observation['pygData'] = pygData

        policy_outputs, agent_state = model(observation, agent_state)
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            graph.reset()
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(
            returns) / len(returns)
    )


class RandomNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm):
        super(RandomNet, self).__init__()
        del observation_shape, use_lstm
        self.num_actions = num_actions
        self.theta = torch.nn.Parameter(torch.zeros(self.num_actions))

    def forward(self, inputs, core_state):
        # print(inputs)
        T, B, *_ = inputs["observation"].shape
        zeros = self.theta * 0
        # set logits to 0
        policy_logits = zeros[None, :].expand(T * B, -1)
        # set baseline to 0
        baseline = policy_logits.sum(dim=1).view(-1, B)

        # sample random action
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1).view(
            T, B
        )
        policy_logits = policy_logits.view(T, B, self.num_actions)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

    def initial_state(self, batch_size):
        return ()


Net = NetHackNet


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
