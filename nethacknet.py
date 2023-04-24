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
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from nle import nethack  # noqa: E402

from gnn import GNN


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


def pygData_from_buffer(env_outputs):

    nodes = env_outputs['pyg_nodes']
    edges = env_outputs['pyg_edges']
    n_nodes = env_outputs['pyg_n_nodes']
    n_edges = env_outputs['pyg_n_edges']

    if len(edges.shape) == 2:
        return Data(nodes, edges)
    elif len(edges.shape) == 3:
        _edges = []
        _nodes = []
        for i in range(nodes.shape[0]):
            _edges.append(edges[i, :n_edges[i], :])
            _nodes.append(nodes[i, :n_nodes[i]])
        pygDatas = [Data(
            x, edge_index) for (x, edge_index) in zip(_nodes, _edges)]
        return Batch.from_data_list(pygDatas)
    elif len(edges.shape) == 4:
        _edges = []
        _nodes = []
        for i in range(nodes.shape[0]):
            for j in range(nodes.shape[1]):
                _samp_edge = edges[i, j, :n_edges[i, j], :]
                if _samp_edge.shape[0] == 0:
                    _samp_edge = _samp_edge.reshape(0)
                _edges.append(_samp_edge)
                _nodes.append(nodes[i, j, :n_nodes[i, j]])
        pygDatas = [Data(
            x=x, edge_index=edge_index) for (x, edge_index) in zip(
            _nodes, _edges)]
        out = Batch.from_data_list(pygDatas)
        return out
    else:
        print(edges.shape)
        assert 0


class NetHackNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super(NetHackNet, self).__init__()

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        # the output dimension of the green blstats MLP and embedding dimension used by the CNNs
        self.k_dim = embedding_dim
        self.h_dim = 512  # the hidden dimension of the LSTM

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        # Red CNN over full glyph map model
        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # Blue CNN crop model
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        # Green blstats MLP
        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        # Our GNN
        self.gnn_out_dim = 16  # HARDCODED
        self.gnn = GNN(in_dim=nethack.MAX_GLYPH+1, out_dim=self.gnn_out_dim)

        # Orange MLP that takes output from (the blue crop CNN, red full glyph map CNN, green blstats MLP, and our GNN) and feeds it to the LSTM or directly to the policy and baseline
        out_dim = self.k_dim  # blstats MLP
        out_dim += self.H * self.W * Y  # CNN over full glyph map
        out_dim += self.crop_dim**2 * Y  # CNN crop model
        out_dim += self.gnn_out_dim  # our GNN

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        # LSTM
        if self.use_lstm:
            self.core = nn.LSTM(self.h_dim, self.h_dim, num_layers=1)

        # Policy and baseline outputs
        self.policy = nn.Linear(self.h_dim, self.num_actions)
        # Harry: tbh I'm not really sure what the baseline is used for
        self.baseline = nn.Linear(self.h_dim, 1)

    def initial_state(self, batch_size=1):  # initial state of the LSTM or empty tuple
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size,
                        self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs, core_state):
        # print("--------- forward -----------")

        # Recover PygData from Nodes, Edges
        pygData = None
        if 'pygData' in env_outputs:
            pygData = env_outputs['pygData']
        else:
            pygData = pygData_from_buffer(env_outputs)

        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # Green blstats MLP
        # -- [B' x F]
        blstats = blstats.view(T * B, -1).float()

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # Blue crop CNN
        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # print("crop", crop)
        # print("at_xy", glyphs[:, coordinates[:, 1].long(), coordinates[:, 0].long()])

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # Red CNN over all glyphs
        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        # Our GNN
        gnn_out = self.gnn(
            x=pygData.x, edge_index=pygData.edge_index, batch=pygData.batch)

        gnn_out = gnn_out.reshape(*glyphs_rep.shape[:-1], self.gnn_out_dim)
        if len(glyphs_rep.shape) > 2:
            print(gnn_out.shape)

        # -- [B x gnn_out_dim]
        reps.append(gnn_out)

        # print("shapes",[x.shape for x in reps])

        # Orange MLP
        try:
            st = torch.cat(reps, dim=1)
        except:
            print("shapes", [x.shape for x in reps])
            st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        # LSTM
        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = st

        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )
