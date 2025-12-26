#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sympy import false

from .trans_face.vit import VisionTransformer
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchGraphConvolution(nn.Module):
    """
    Graph convolutional layers supporting batch processing (compatible with sparse adjacency matrices)
    Input:
    -x: (B, n, d_in) node feature
    -adj: (B, n, n) adjacency matrix (supports weighted/sparse)
    Output
    - (B, n, d_out) updated node features
    """
    def __init__(self, in_features, out_features,
                 use_degree_norm=True,
                 residual=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.use_degree_norm = use_degree_norm
        self.residual = residual

        # Initialize parameters
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        # 1. Feature transformation
        h = self.linear(x)  # (B, n, d_out)

        # 2. Degree matrix normalization (optional)
        if self.use_degree_norm:
            degree = adj.sum(dim=2, keepdim=True) + 1e-8  # Prevent division by zero
            h = h / degree.clamp_min(1.0)  # Ensure that the minimum degree is 1

        # 3. Graph propagation + residual join
        out = torch.bmm(adj, h)  # Batch processing matrix multiplication
        if self.residual:
            out = out + x[:, :, :out.size(2)]  # Dimension alignment

        return self.activation(out)

class DecoderRNN(nn.Module):
    def __init__(self, cnn_embed_dim=300, h_rnn_layers=3, h_rnn=256, h_fc_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = cnn_embed_dim
        self.h_RNN_layers = h_rnn_layers  # RNN hidden layers
        self.h_RNN = h_rnn  # RNN hidden nodes
        self.h_FC_dim = h_fc_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_rnn_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        # 3. Spatio-temporal attention
        self.attention = nn.Sequential(
            nn.Linear(self.h_RNN, self.h_RNN // 2),
            nn.Tanh(),
            nn.Linear(self.h_RNN // 2, 1, bias=False)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_rnn):
        self.LSTM.flatten_parameters()
        rnn_out, (h_n, h_c) = self.LSTM(x_rnn, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        attention_out =  F.softmax(self.attention(rnn_out),  dim=1)
        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x, attention_out



class TransFaceGCNEmotion(nn.Module):
    def __init__(self, image_size=112, cnn_embed_dim=512, num_classes=2, emotion_num=7, frame=32,h_rnn_layers=3,gcn_layers=2):
        super().__init__()
        self.trans_face_model = VisionTransformer(
            img_size=image_size, patch_size=9, num_classes=cnn_embed_dim, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
        #self.trans_face_model.load_state_dict(torch.load('./glint360k_model_TransFace_S.pt'))
        self.rnn_decoder = DecoderRNN(cnn_embed_dim=cnn_embed_dim,h_rnn_layers=h_rnn_layers, num_classes=num_classes)
        self.emotion_mlp = nn.Sequential(
            nn.Linear(cnn_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, emotion_num),
        )

        self.gcn_conv = nn.ModuleList(
            [
                BatchGraphConvolution(in_features=emotion_num, out_features=emotion_num,
                                      residual=True)
                for i in range(gcn_layers)]
        )
        self.cls_head = nn.Sequential(
            nn.Linear(frame * emotion_num, frame * emotion_num),
            nn.LayerNorm(frame * emotion_num),
            nn.ReLU(),
            nn.Linear(frame * emotion_num, frame * emotion_num // 2),
            nn.Linear(frame * emotion_num // 2, num_classes),
        )

    def batch_cosine_similarity(self,x):
        """Calculate the cosine similarity of all node pairs within a batch (vectorized implementation)"""
        # x: (B, N, D) = (B, 32, 7)
        x_norm = x / (torch.norm(x, dim=2, keepdim=True) + 1e-8)  # Normalization
        return torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

    def build_batch_adjacency(self,features, top_k=None, threshold=None):
        """
            Parameter
            features: Batch feature tensor of (B, 32, 7)
            top_k: The maximum number of connections retained per node (if None, all connections will be retained)
            threshold: Similarity threshold (set to 0 for connections less than this value)
            Return
            The adjacency matrix of (B, 32, 32)
        """
        with torch.no_grad():
            adj = self.batch_cosine_similarity(features)

            # Sparsification processing
            if top_k is not None:
                values, indices = torch.topk(adj, k=top_k, dim=2)
                adj = torch.zeros_like(adj).scatter_(2, indices, values)
            if threshold is not None:
                adj[adj < threshold] = 0

            return adj.clamp(-1, 1)  # Ensure the numerical range

    def forward(self, x_3d,is_train=false):
        emotion_seq = []
        cnn_embed_seq = []
        # Extract the facial representations of each frame
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                local_embeddings, weight, local_patch_entropy = self.trans_face_model(x_3d[:, t, :, :, :])
            # Generate emoji category embeddings
            cnn_embed_seq.append(local_embeddings)
            emotion_seq.append(self.emotion_mlp(local_embeddings))
        emotion_out = torch.stack(emotion_seq, dim=0).transpose_(0, 1)

        feature = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        out, attention_out = self.rnn_decoder(feature)

        # Graph convolution is performed on the embedding of expression categories to construct the relationship
        adj = self.build_batch_adjacency(emotion_out)
        gcn_feature = torch.softmax(emotion_out, dim=-1)
        for i, blk in enumerate(self.gcn_conv):
            gcn_feature = blk(gcn_feature, adj)
        gcn_feature = gcn_feature * attention_out
        if is_train:
            return self.cls_head(gcn_feature.flatten(1)), emotion_out, gcn_feature
        else:
            return self.cls_head(gcn_feature.flatten(1))

if __name__ == '__main__':
    model = TransFaceGCNEmotion(image_size=112)
    video = torch.rand(16, 32, 3, 112, 112)  # (B,T,C,H,W)
    output = model(video)  # (B,1)
    flops, params = get_model_complexity_info(model, (32, 3, 112, 112), as_strings=True, print_per_layer_stat=False)
    print('All Flops:  ' + flops)
    print('All Params: ' + params)

    flops, params = get_model_complexity_info(model.trans_face_model, (3, 112, 112), as_strings=True, print_per_layer_stat=False)
    print('Trans_face_model Flops:  ' + flops)
    print('Trans_face_model Params: ' + params)


