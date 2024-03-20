# import torch as T
# import torch.nn as nn
# from torch.nn import functional as F
#
# __all__ = ['DSN']
#
# class DSN(nn.Module):
#     """ Deep Summarization Network """
#
#     def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
#         super(DSN, self).__init__()
#         assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru"
#
#         if cell == 'lstm':
#             print("lstmmmmmm")
#             self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
#         elif cell == 'gru':
#             self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
#
#         self.fc = nn.Linear(hid_dim*2, 1)
#
#     def forward(self, x):
#         h, _ = self.rnn(x)
#         p = F.sigmoid(self.fc(h))
#
#         return p


# new transformer decoder trying
import torch.nn.functional as F
import torch.nn as nn
import torch
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        x = x + self.dropout(self.linear2(F.relu(self.linear1(x))))
        x = self.norm2(x)
        return x

class DSN(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=256, num_layers=1,cell="hey", nhead=8, num_decoder_layers=6):
        super(DSN, self).__init__()

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=in_dim, nhead=nhead) for _ in range(num_decoder_layers)]
        )
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        p = torch.sigmoid(self.fc(x))
        return p





# # Original transformer decoder always used
# import torch.nn.functional as F
# import torch.nn as nn
# import torch
#
#
# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#     def forward(self, x):
#         attn_output, _ = self.self_attn(x, x, x)
#         x = x + attn_output
#         x = self.norm1(x)
#         x = self.linear2(self.dropout(F.relu(self.linear1(x))))
#         x = x + attn_output
#         x = self.norm2(x)
#         return x
#
# class DSN(nn.Module):
#     def __init__(self, in_dim=2048, hid_dim=256, num_layers=1,cell="hey", nhead=8, num_decoder_layers=6):
#         super(DSN, self).__init__()
#
#         self.transformer_layers = nn.ModuleList(
#             [TransformerDecoderLayer(d_model=in_dim, nhead=nhead) for _ in range(num_decoder_layers)]
#         )
#         self.fc = nn.Linear(in_dim, 1)
#
#     def forward(self, x):
#         for layer in self.transformer_layers:
#             x = layer(x)
#         # Aggregate the output over time steps (you can experiment with other aggregation methods)
#         # x = x.mean(dim=1)
#         p = torch.sigmoid(self.fc(x))
#         return p









# import torch
# import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
# class DSN(nn.Module):
#     def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, dropout=0.5, nhead=4):
#         super(DSN, self).__init__()
#
#         encoder_layers = TransformerEncoderLayer(d_model=in_dim, nhead=nhead)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc = nn.Linear(in_dim, 1)
#
#     def forward(self, x):
#         # Assuming x has shape (batch_size, sequence_length, feature_dim)
#         batch_size, sequence_length, feature_dim = x.size()
#
#         x = self.transformer_encoder(x)
#         x_last = x[:, -1, :]  # Take the output of the last position in the sequence
#
#         x_last = self.dropout(x_last)
#         p = torch.sigmoid(self.fc(x_last))
#
#         return x_last


#This is original transformer
# import torch
# import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
# class DSN(nn.Module):
#     def __init__(self, in_dim=2048, hid_dim=256, num_layers=1, dropout=0.5,cell="hey"):
#         super(DSN, self).__init__()
#
#         encoder_layers = TransformerEncoderLayer(d_model=in_dim, nhead=4)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc = nn.Linear(in_dim, 1)
#
#     def forward(self, x):
#         x = self.transformer_encoder(x)
#         x = torch.mean(x, dim=1)  # Global average pooling
#
#         x = self.dropout(x)
#         p = torch.sigmoid(self.fc(x))
#
#         return x




#This code i tried to use lstm model parameters to set the transformer model parameters
# import torch
# import torch.nn as nn
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
#
# class DSN(nn.Module):
#     def __init__(self, in_dim=2048, hid_dim=256, num_layers=1, dropout=0.5):
#         super(DSN, self).__init__()
#
#         # Shared layers (modify based on your actual architecture)
#         shared_hid_dim = 512  # Define the shared hidden dimension
#         self.shared_layers = nn.Sequential(
#             nn.Linear(in_dim, shared_hid_dim),
#             nn.ReLU(),
#             # Add more shared layers if needed
#         )
#
#         # Transformer-specific layers
#         encoder_layers = TransformerEncoderLayer(d_model=shared_hid_dim, nhead=4)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.fc = nn.Linear(shared_hid_dim, 1)
#
#     def forward(self, x):
#         x_shared = self.shared_layers(x)
#         x_transformer = self.transformer_encoder(x_shared)
#         x = torch.mean(x_transformer, dim=1)  # Global average pooling
#
#         x = self.dropout(x)
#         p = torch.sigmoid(self.fc(x))
#
#         return x

# Load the shared weights from the RNN checkpoint


# Continue with fine-tuning or other modifications as needed

