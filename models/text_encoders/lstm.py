import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.abc import AbstractBaseTextEncoder
from models.text_encoders.utils import retrieve_last_timestamp_output
from transformers import BertModel, BertConfig
from transformers import AutoModel
from transformers import AutoTokenizer

class SimpleLSTMEncoder(AbstractBaseTextEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        word_embedding_size = kwargs.get('word_embedding_size', 512)
        lstm_hidden_size = kwargs.get('lstm_hidden_size', 512)
        feature_size = feature_size

        self.embedding_layer = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Dropout(p=0.1),
            nn.Linear(lstm_hidden_size, feature_size),
        )
        self.bn = nn.BatchNorm1d(feature_size)

    def forward(self, x, lengths):
        # x is a tensor that has shape of (batch_size * seq_len)
        x = self.embedding_layer(x)  # x's shape (batch_size * seq_len * word_embed_dim)
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(x)
        outputs = retrieve_last_timestamp_output(lstm_outputs, lengths)

        outputs = self.fc(outputs)
        return outputs

    @classmethod
    def code(cls) -> str:
        return 'lstm'

class BertEncoder(nn.Module):
    def __init__(self, feature_size, mode):
        super().__init__()
        model_name = 'roberta-base'
        self.mode = mode
        assert mode in ['train', 'eval']
        #self.model = BertModel.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = 768 if model_name == 'roberta-base' else 1024
        self.feature_size = feature_size
        #self.pool= nn.AdaptiveAvgPool2d((1, hidden_size))
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, feature_size),
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x, attn_mask):
        #token = self.tokenizer.batch_encode_plus(x, padding='longest',
        #                                    return_tensors='pt')
        if self.mode == 'train':
            self.model.train()
            x = self.model(x,attn_mask).last_hidden_state
        else:
            self.model.eval()
            with torch.no_grad():
                x = self.model(x).last_hidden_state
        #x = torch.max(x, dim=1)[0]
        #x = torch.mean(x, dim=1)
        #x = self.fc(x)

        return x

    def code() -> str:
        return 'bert'

class NormalizationLSTMEncoder(SimpleLSTMEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        self.norm_scale = kwargs['norm_scale']

    def forward(self, x: torch.Tensor, lengths: torch.LongTensor) -> torch.Tensor:
        outputs = super().forward(x, lengths)
        return F.normalize(outputs) * self.norm_scale

    @classmethod
    def code(cls) -> str:
        return 'norm_lstm'


class SimplerLSTMEncoder(AbstractBaseTextEncoder):
    def __init__(self, vocabulary_len, padding_idx, feature_size, *args, **kwargs):
        super().__init__(vocabulary_len, padding_idx, feature_size, *args, **kwargs)
        word_embedding_size = kwargs.get('word_embedding_size', 512)

        self.embedding_layer = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(word_embedding_size, self.feature_size, batch_first=True)

    def forward(self, x, lengths):
        # x is a tensor that has shape of (batch_size * seq_len)
        x = self.embedding_layer(x)  # x's shape (batch_size * seq_len * word_embed_dim)
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(x)
        return retrieve_last_timestamp_output(lstm_outputs, lengths)

    @classmethod
    def code(cls) -> str:
        return 'simpler_lstm'
