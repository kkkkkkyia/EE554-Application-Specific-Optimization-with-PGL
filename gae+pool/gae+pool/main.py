import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from typing import List
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sklearn
from torch_geometric.nn import GAE
import argparse
from torch_geometric.utils import negative_sampling
import torch_geometric.utils as utils
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
# import models
from tqdm import tqdm
import json
from diff_pool import diff_pool_net
from mincut_pool import mincut_net
from dmon_pool import dmon_net

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    help="The name of the device, can be 'cpu', 'cuda', 'mps'",
    default="cpu",
    dest="DEVICE",
    type=str,
)
parser.add_argument(
    "--data-directory",
    help="The directory of the GEXF computation graphs",
    default="./gae+pool/dataset",
    dest="DATA_DIRECTORY",
    type=str,
)
parser.add_argument("--random-state", default=42, dest="RANDOM_STATE", type=int)
parser.add_argument(
    "--gae-epochs",
    default=100,
    help="Number of epochs for the auto-encoder",
    dest="GAE_EPOCHS",
    type=int,
)
parser.add_argument("--test-size", default=0.1, dest="TEST_SIZE", type=float)
parser.add_argument("--train-batch-size", default=1, dest="TRAIN_BATCH_SIZE", type=int)
parser.add_argument("--test-batch-size", default=128, dest="TEST_BATCH_SIZE", type=int)
parser.add_argument(
    "--encoder-type-embedding-size",
    default=32,
    dest="ENCODER_TYPE_EMBEDDING_SIZE",
    type=int,
)
parser.add_argument(
    "--encoder-output-size",
    default=16,
    dest="ENCODER_OUTPUT_SIZE",
    type=int,
)
parser.add_argument(
    "--number-of-clusters",
    default=2,
    dest="NUMBER_OF_CLUSTERS",
    type=int,
)

args = parser.parse_args()
print(args)
print("-------------------")


def get_directory_gexf_filenames(
    dirname: Path = Path("./dataset"),
) -> List[str]:
    """Get the gexf files in a given directory.

    :param dirname: the name of the directory containing the GEXF files
    :return: a list of gexf filenames
    """
    return [file for file in os.listdir(dirname) if file.endswith("gexf")]


def load_data(
    dirname: Path = Path("./gae+pool/dataset"),
) -> List[nx.classes.digraph.DiGraph]:
    """Load the dataset with the computation graphs.

    Read the computation graphs. Each sample of the dataset corresponds to
    a computation graph. The nodes of each computation graph are enriched
    with a list of features.

    :param dirname: the name of the directory containing the GEXF files
    :return: the computation graphs as a list of networkx objects
    """
    computation_graphs = []
    gexf_filenames = get_directory_gexf_filenames()
    for gfile in gexf_filenames:
        try:
            print(f"Loading {gfile} file...")
            filepath = os.path.join(dirname, gfile)
            print(filepath)
            G = nx.read_gexf(filepath)
            computation_graphs.append(G)
        except:
            pass

    return computation_graphs



def extract_features(
    computation_graphs: List[nx.classes.digraph.DiGraph],
    le: sklearn.preprocessing._label.LabelEncoder,
) -> List[nx.classes.digraph.DiGraph]:
    #Extract features for each node and edge of the computation graphs.

    #Note that the changes are applied to the original computation graphs implicitly.

    #:param computation_graphs: the list of networkx object of computation graphs
    #:return: a list of networkx object enriched
    
    # Learn the unique types of nodes.
    unique_node_types = set()
    for cg in computation_graphs:
        unique_node_types = unique_node_types | set(
            nx.get_node_attributes(cg, "type").values()
        )
    unique_node_types = list(unique_node_types)
    le.fit(unique_node_types)

    # Convert the types of nodes to numbers.
    for cg in computation_graphs:
        node_types = list(
            le.transform([cg.nodes[node_id]["type"] for node_id in cg.nodes()])
        )
        for node_id, encoded_type in zip(cg.nodes(), node_types):
            cg.nodes[node_id]["type"] = encoded_type

    for cg in computation_graphs:
        for node_id in cg.nodes():
            # FIXME: Calculate the correct features.
            cg.nodes[node_id]["x"] = np.array(
                [cg.nodes[node_id]["type"]], dtype=np.float32
            )

    return computation_graphs



"""

def extract_features(
        computation_graphs: List[nx.classes.digraph.DiGraph],
        le: sklearn.preprocessing._label.LabelEncoder,
) -> List[nx.classes.digraph.DiGraph]:
    #Extract features for each node and edge of the computation graphs.

    #Note that the changes are applied to the original computation graphs implicitly.

    #:param computation_graphs: the list of networkx object of computation graphs
    #:return: a list of networkx object enriched 

    # Attempt to learn the unique types of nodes. If none exist, use a default placeholder.
    unique_node_types = set()
    for cg in computation_graphs:
        node_types = nx.get_node_attributes(cg, "type").values()
        if node_types:
            unique_node_types.update(node_types)

    if not unique_node_types:
        print("Warning: No node types found in any of the graphs. Using default placeholder.")
        unique_node_types = {'default'}  # Add a default type to avoid errors in label encoding

    # Fit the LabelEncoder to the unique node types found or the default placeholder
    le.fit(list(unique_node_types))

    # Assign a default type index for graphs that don't have a 'type' attribute
    default_type_index = list(le.classes_).index('default')

    # Convert the types of nodes to numbers, or use the default type index.
    for cg in computation_graphs:
        node_types = {}
        for node_id, node_data in cg.nodes(data=True):
            node_type = node_data.get("type", "default")  # Use the default if no type found
            node_types[node_id] = le.transform([node_type])[0]

        # Update node features in the graph
        nx.set_node_attributes(cg, node_types, 'x')

    return computation_graphs

"""


#def convert_networkx_graphs_to_data(
#    computation_graphs: List[nx.classes.digraph.DiGraph],
#) -> List[Data]:
#    return [from_networkx(cg) for cg in computation_graphs]

def convert_networkx_graphs_to_data(computation_graphs):
    data_list = []
    for graph in computation_graphs:
        if not graph.nodes():  # Check if the graph has no nodes
            continue  # Skip processing this graph

        data = from_networkx(graph)  # Convert from NetworkX to PyG Data

        # Handle node attributes safely
        for node_id, node_data in graph.nodes(data=True):
            if 'y' in node_data:
                if hasattr(data, 'y'):
                    data.y = torch.cat([data.y, torch.tensor([node_data['y']], dtype=torch.long)])
                else:
                    data.y = torch.tensor([node_data['y']], dtype=torch.long)

        data_list.append(data)
    return data_list


"""
def convert_networkx_graphs_to_data(computation_graphs):
    data_list = []
    for graph in computation_graphs:
        # Assuming you already have a way to extract x and edge_index from graph
        data = from_networkx(graph)
        data.x = torch.tensor([...], dtype=torch.float)  # Your existing node features
        data.edge_index = ...  # Your edge index setup

        # Check if 'type' exists and add it to data
        if 'type' in graph.graph:
            data.type = torch.tensor([graph.graph['type']], dtype=torch.float32)
        else:
            data.type = torch.tensor([-1], dtype=torch.float32)  # Default or placeholder

        data_list.append(data)

    return data_list
"""

# Load the data.
print("######## Loading data ########")
computation_graphs = load_data(args.DATA_DIRECTORY)
# Extract initial features.
print("######## Extracting initial features ########")
le = LabelEncoder()
computation_graphs = extract_features(computation_graphs, le)
# Convert networkx to Data objects.
print("######## Converting networkx to Data objects ########")
computation_graphs = convert_networkx_graphs_to_data(computation_graphs)
# Split data to train and test.
train_computation_graphs, test_computation_graphs = train_test_split(
    computation_graphs, test_size=args.TEST_SIZE, random_state=args.RANDOM_STATE
)
# Create a dataloader object for each split.
train_loader = DataLoader(
    train_computation_graphs, batch_size=args.TRAIN_BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_computation_graphs, batch_size=args.TEST_BATCH_SIZE, shuffle=False
)

device = torch.device(args.DEVICE)

##################################
# Representation learning phase.#
#################################
in_channels, out_channels = args.ENCODER_TYPE_EMBEDDING_SIZE, args.ENCODER_OUTPUT_SIZE
number_of_unique_types = le.classes_.shape[0]

# gae_model = GAE(
#     models.GCNEncoder(
#         in_channels,
#         out_channels,
#         number_of_unique_types,
#         args.ENCODER_TYPE_EMBEDDING_SIZE,
#     )
# )

# FIXME: Need to determine the number of hidden layer and number of cluster in the end
# Question: What is the relationship between args.ENCODER_TYPE_EMBEDDING_SIZE and in_channel

max_nodes = 100
print(f"in_chanel {in_channels}, out_chanel {out_channels}")
gae_model = GAE(
    diff_pool_net(
        in_channels,
        out_channels,
        max_nodes,
        number_of_unique_types,
        args.ENCODER_TYPE_EMBEDDING_SIZE,
    )
)

# gae_model = GAE(
#     mincut_net(
#         in_channels,
#         out_channels,
#         max_nodes,
#         number_of_unique_types,
#         args.ENCODER_TYPE_EMBEDDING_SIZE,
#     )
# )

# gae_model = GAE(
#     dmon_net(
#         in_channels,
#         out_channels,
#         max_nodes,
#         number_of_unique_types,
#         args.ENCODER_TYPE_EMBEDDING_SIZE,
#     )
# )
gae_model = gae_model.to(device)
gae_optimizer = torch.optim.Adam(gae_model.parameters(), lr=0.01)


def train_gae(train_data):
    """Train a graph auto encoder on link prediction."""
    gae_model.train()
    gae_optimizer.zero_grad()
    z = gae_model.encode(train_data.x, train_data.edge_index, train_data.type,one_hot=False)
    negative_edge_index = negative_sampling(train_data.edge_index)
    z = z.squeeze()

    loss = gae_model.recon_loss(z, train_data.edge_index, negative_edge_index)

    loss.backward()
    gae_optimizer.step()
    return loss


@torch.no_grad()
def test_gae(test_loader: torch_geometric.loader.dataloader.DataLoader):
    gae_model.eval()
    y_true = []
    y_pred = []
    for data in test_loader:
        z = gae_model.encode(data.x, data.edge_index, data.type)
        z = z.squeeze()
        negative_edge_index = negative_sampling(data.edge_index)

        pos_y = z.new_ones(data.edge_index.size(1))
        neg_y = z.new_zeros(negative_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = gae_model.decode(z, data.edge_index)
        neg_pred = gae_model.decode(z, negative_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        y_true.extend(y)
        y_pred.extend(pred)
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


for epoch in range(1, args.GAE_EPOCHS + 1):
    average_train_loss = 0.0
    for data in train_loader:
        loss = train_gae(data)
        average_train_loss += loss.item()
    average_train_loss /= len(train_loader)
    auc, ap = test_gae(test_loader)
    print(
        f"Epoch: {epoch:03d}, train loss {average_train_loss:.4f} test AUC: {auc:.4f}, test AP: {ap:.4f}"
    )

#def get_gae_node_embeddings(computation_graphs):
#    gae_node_embeddings = []
#    for cg in computation_graphs:
#        z = gae_model.encode(cg.x, cg.edge_index, cg.type)
#        gae_node_embeddings.append(z.detach().numpy())
#    return gae_node_embeddings

def get_gae_node_embeddings(computation_graphs):
    gae_node_embeddings = []
    for cg in computation_graphs:
        z = gae_model.encode(cg.x, cg.edge_index, cg.type)
        # Ensure that z is reshaped to 2D if necessary, flattening all dimensions except the last
        z_reshaped = z.detach().numpy().reshape(-1, z.shape[-1])
        gae_node_embeddings.append(z_reshaped)
    return gae_node_embeddings



train_gae_node_embeddings = get_gae_node_embeddings(train_computation_graphs)
test_gae_node_embeddings = get_gae_node_embeddings(test_computation_graphs)
all_gae_node_embeddings = get_gae_node_embeddings(computation_graphs)

#############################
# Graph partitioning phase.#
############################
print("Calculating cluster for the nodes for each computation graph...")
from sklearn.cluster import SpectralClustering

print("Training set...")
#train_node_cluster_labels = []
#for node_emb in tqdm(train_gae_node_embeddings):
#    clustering = SpectralClustering(
#        n_clusters=args.NUMBER_OF_CLUSTERS, assign_labels="discretize", random_state=42
#    ).fit(node_emb)
#    train_node_cluster_labels.append(clustering.labels_)
# Clustering phase where you use SpectralClustering on the embeddings
train_node_cluster_labels = []
for node_emb in tqdm(train_gae_node_embeddings):
    if node_emb.ndim > 2:  # Check if the data is more than 2D
        node_emb = node_emb.reshape(node_emb.shape[0], -1)  # Flatten to 2D
    clustering = SpectralClustering(
        n_clusters=args.NUMBER_OF_CLUSTERS, assign_labels="discretize", random_state=42
    ).fit(node_emb)
    train_node_cluster_labels.append(clustering.labels_)



print("Test set...")
test_node_cluster_labels = []
for node_emb in tqdm(test_gae_node_embeddings):
    clustering = SpectralClustering(
        n_clusters=args.NUMBER_OF_CLUSTERS, assign_labels="discretize", random_state=42
    ).fit(node_emb)
    test_node_cluster_labels.append(clustering.labels_)


print("All of the computation graphs...")
all_node_cluster_labels = []
for node_emb in tqdm(all_gae_node_embeddings):
    clustering = SpectralClustering(
        n_clusters=args.NUMBER_OF_CLUSTERS, assign_labels="discretize", random_state=42
    ).fit(node_emb)
    all_node_cluster_labels.append(clustering.labels_)


##################
# Model training.#
##################


# FIXME: We should call OpenVINO at this point to find the correct device for each node.
def assign_labels_to_nodes(
    computation_graphs: List[nx.classes.digraph.DiGraph],
    node_cluster_labels: List[np.ndarray],
):
    """Assign labels to the nodes based on the node clusters.

    :param computation_graphs: the name of the directory containing the GEXF files
    :return: the computation graphs as a list of networkx objects
    """
    for cg, nc in zip(computation_graphs, node_cluster_labels):
        cg.y = nc

    return computation_graphs


train_computation_graphs = assign_labels_to_nodes(
    train_computation_graphs, train_node_cluster_labels
)
test_computation_graphs = assign_labels_to_nodes(
    test_computation_graphs, test_node_cluster_labels
)
computation_graphs = assign_labels_to_nodes(computation_graphs, all_node_cluster_labels)

gexf_filenames = get_directory_gexf_filenames()

def save_graph_data(computation_graphs, json_dirpath):
    json_dirpath = Path(json_dirpath)
    json_dirpath.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    for index, graph in enumerate(computation_graphs):
        # Assuming each graph's nodes have labels and cluster IDs stored in 'y'
        # and that you may have stored additional properties in 'data' if necessary
        node_clusters = []
        for node_id, node_data in enumerate(graph.y.tolist()):
            node_clusters.append({
                "node_name": f"Node{node_id}",
                "node_id": node_id,
                "subgraph_id": str(node_data)
            })

        # Save to JSON file
        json_filename = f"graph_{index}.json"
        node_clusters_filepath = json_dirpath / json_filename
        with open(node_clusters_filepath, "w") as file:
            json.dump(node_clusters, file, indent=2)

# Call the function with paths and graphs
save_graph_data(computation_graphs, "./gae+pool/dataset/node_clusters")


# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F

# class GCNNodeClassifier(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GCNNodeClassifier, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         return F.log_softmax(x, dim=1)

# # Define the model, optimizer, and loss function

# num_features = 10
# hidden_channels = 16
# out_channels = 20
# model = GCNNodeClassifier(in_channels=num_features, hidden_channels=hidden_channels, out_channels=out_channels)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.NLLLoss


# # Training loop
# def train():
#     model.train()
#     optimizer.zero_grad()
#     output = model(data)
#     loss = criterion(output[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()

# # Testing loop
# def test():
#     model.eval()
#     output = model(data)
#     pred = output.argmax(dim=1)
#     correct = pred[data.test_mask] == data.y[data.test_mask]
#     acc = correct.sum().item() / data.test_mask.sum().item()
#     return acc

# # Training and testing
# for epoch in range(200):
#     train()
#     acc = test()
#     print(f'Epoch: {epoch + 1}, Test Accuracy: {acc:.4f}')
