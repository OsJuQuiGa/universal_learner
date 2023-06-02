from typing import Optional, Any
import torch
import time
from torch import Tensor
from torch.nn import LayerNorm

LAST_ATTN = True

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        ''' Positional encoding used as temporal encoding
        '''
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None, rattn: bool = False) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        if rattn:
            attn = src2.detach().clone()
        else:
            attn = None
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
            src_key_padding_mask: Optional[Tensor] = None, rattn:bool = False) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        if rattn:
            attn = src2.detach().clone()
        else:
            attn = None
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerDecoder(torch.nn.TransformerDecoder):        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = None
        self.only_last_attn = LAST_ATTN

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        attn = []
        for i, mod in enumerate(self.layers):
            if self.only_last_attn:
                rattn = i == len(self.layers) - 1
            else:
                rattn = True
            output, attn_a = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         rattn = rattn)
            attn.append(attn_a)
        if self.only_last_attn:
            attn = attn[-1]
        else:
            attn = torch.mean(torch.stack(attn, dim=0), dim=0)
        self.attn = attn

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = None
        self.only_last_attn = LAST_ATTN
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, 
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        attn = []
        for i, mod in enumerate(self.layers):
            if self.only_last_attn:
                rattn = i == len(self.layers) - 1
            else:
                rattn = True
            tmp_out = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, rattn = rattn)
            output, attn_a = tmp_out
            attn.append(attn_a)
        if self.only_last_attn:
            attn = attn[-1]
        else:
            attn = torch.mean(torch.stack(attn, dim=0), dim=0)
        self.attn = attn

        if self.norm is not None:
            output = self.norm(output)
        return output

class AttnTransformer(torch.nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None, seq_len = 1, attn='all') -> None:
        if custom_encoder is None and attn == 'all' or attn == 'encoder':
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            custom_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        if custom_decoder is None and attn == 'all' or attn == 'decoder':
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            custom_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
            dropout, activation, custom_encoder, custom_decoder
        )
        self.seq_len = seq_len
        self.enc_attn = None
        self.dec_attn = None
    def get_out_dim(self): #no batch
        return (self.seq_len, self.d_model)

    def forward(self, x):
        x = super().forward(x)
        enc_attn = self.encoder.attn
        dec_attn = self.decoder.attn
        return x, enc_attn, dec_attn


