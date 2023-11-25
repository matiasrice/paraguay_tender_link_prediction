from torch_geometric.nn import GATConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
import torch
import tqdm
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()

        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Add dropout for regularization
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        prediction = (edge_feat_user * edge_feat_movie).sum(dim=-1)
    
        prediction = torch.sigmoid(prediction)

        return prediction

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, data, large_features):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.user_lin = torch.nn.Linear(25, hidden_channels)
        if not large_features:
            self.movie_lin = torch.nn.Linear(25, hidden_channels)
        else:
            self.movie_lin = torch.nn.Linear(34, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["institution"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["supplier"].num_nodes, hidden_channels)

        # Instantiate heterogeneous GNN with GATConv
        self.gnn = GNN(hidden_channels, num_heads)

        # Convert GNN model into a heterogeneous variant
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          #"institution": self.user_emb(data["institution"].node_id),
          "institution": self.user_lin(data["institution"].x) + self.user_emb(data["institution"].node_id),
          "supplier": self.movie_lin(data["supplier"].x) + self.movie_emb(data["supplier"].node_id),
        } 

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["institution"],
            x_dict["supplier"],
            data["institution", "rates", "supplier"].edge_label_index,
        )

        return pred


def gat_train(model, sampled_data, train_loader, add_negative_samples=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 5):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data["institution", "rates", "supplier"].edge_label
            if add_negative_samples:
                # For negative samples, use the labels indicating whether the edge exists or not
                neg_edge_label = 1 - ground_truth
                ground_truth = torch.cat([ground_truth, neg_edge_label], dim=0)

            ground_truth = ground_truth[:pred.size(0)]
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        
    return model


