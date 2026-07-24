# GNN quasi identique à celui de Brandstetter et al. (MP-PDE-Solver), mais
# sans torch_cluster : la grille 1D étant fixe pour tous les échantillons,
# edge_index est construit une fois à la main plutôt que via radius_graph.
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def build_chain_edges(n_nodes: int, k: int) -> torch.Tensor:
    # Chaque nœud relié à ses k voisins directs de chaque côté (arêtes dans
    # les deux sens). Empilé sur plusieurs couches de message passing, le
    # champ réceptif effectif grandit de k nœuds par couche.
    src, dst = [], []
    for i in range(n_nodes):
        for off in range(1, k + 1):
            j = i + off
            if j < n_nodes:
                src += [i, j]
                dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long)


def build_batched_edges(base_edge_index: torch.Tensor, n_nodes: int, b: int):
    # b graphes identiques (même topologie 1D) empilés en un seul grand
    # graphe déconnecté, comme le fait torch_geometric.data.Batch.
    offsets = torch.arange(b, device=base_edge_index.device).repeat_interleave(base_edge_index.shape[1]) * n_nodes
    edge_index = base_edge_index.repeat(1, b) + offsets
    batch = torch.arange(b, device=base_edge_index.device).repeat_interleave(n_nodes)
    return edge_index, batch


class GNN_Layer(MessagePassing):
    """Message passing layer (formules 8-9 de Brandstetter et al. 2022)."""

    def __init__(self, in_features: int, out_features: int, hidden_features: int,
                 time_window: int, n_variables: int):
        super().__init__(node_dim=-2, aggr="mean")
        self.in_features = in_features
        self.out_features = out_features

        self.message_net_1 = nn.Sequential(
            nn.Linear(2 * in_features + time_window + 1 + n_variables, hidden_features), Swish())
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), Swish())
        self.update_net_1 = nn.Sequential(
            nn.Linear(in_features + hidden_features + n_variables, hidden_features), Swish())
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), Swish())
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        return self.norm(x, batch)

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        m = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        return self.message_net_2(m)

    def update(self, message, x, variables):
        upd = self.update_net_2(self.update_net_1(torch.cat((x, message, variables), dim=-1)))
        return x + upd if self.in_features == self.out_features else upd


class WaveGNN(nn.Module):
    """
    Entrée par nœud : [u(t), u(t-ndt), ..., pos_x, pos_t, A_norm, omega_norm]
    (m_back valeurs brutes d'historique + position + variables globales,
    façon Brandstetter -- pas le stencil U/Ut/Uxx du MLP existant).
    forward(X) -> Y a la même signature que commun.Reseau : X de forme
    (n_nodes * b, n_features), Y de forme (n_nodes * b, n_fwd).
    """

    def __init__(self, n_nodes: int, m_back: int, n_fwd: int, n_eq_vars: int = 2,
                 hidden_features: int = 32, hidden_layer: int = 5, k: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.m_back = m_back
        self.n_fwd = n_fwd
        n_variables = 1 + n_eq_vars  # pos_t + (A_norm, omega_norm)

        self.register_buffer("base_edges", build_chain_edges(n_nodes, k), persistent=False)

        self.embedding_mlp = nn.Sequential(
            nn.Linear(m_back + 1 + n_variables, hidden_features), Swish(),
            nn.Linear(hidden_features, hidden_features), Swish())

        self.gnn_layers = nn.ModuleList(
            GNN_Layer(hidden_features, hidden_features, hidden_features, m_back, n_variables)
            for _ in range(hidden_layer))

        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features), Swish(),
            nn.Linear(hidden_features, n_fwd))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.shape[0] % self.n_nodes == 0, (
            f"X doit contenir un multiple de n_nodes={self.n_nodes} lignes (reçu {X.shape[0]})")
        b = X.shape[0] // self.n_nodes

        u = X[:, :self.m_back]
        pos_x = X[:, self.m_back:self.m_back + 1]
        variables = X[:, self.m_back + 1:]

        edge_index, batch = build_batched_edges(self.base_edges, self.n_nodes, b)

        h = self.embedding_mlp(torch.cat((u, pos_x, variables), dim=-1))
        for layer in self.gnn_layers:
            h = layer(h, u, pos_x, variables, edge_index, batch)

        return self.decoder_mlp(h)
