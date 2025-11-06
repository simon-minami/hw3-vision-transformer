# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
       
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Initialize the following layers and parameters to perform attention
        # This class assumes that the input dimension for query, key and value is embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)
            
    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape
        # N,S,D is batch, query seqence length, embedding dim
        # N,T,D is similar idea
        # is self attention, S=T cuz queries, keys come from same sequence
        # T is always same for key and value because they need to be in corresonding pairs
        # queries represent things doing the attending
        # values tells you if something attends to me, heres the info they get
        # so you need same number of keys and values
        # in crosss attention you could have queries come from image caption
        # keys/values could come from different sequence like image patches
        # so S doesn't have to equal T
       
        # TODO : Compute attention 
    
        #project query, key and value  - 
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # shape remains the same N, S, D for queries
        # N, T, D for keys and values
        # 

        #compute dot-product attention. Don't forget the scaling value!
        #Expected shape of dot_product is (N, S, T)
        # torch automatically does batched matrix multiplcation
        # divide by the d_model
        dot_product = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(D)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            # we want these to be zeroed out POST softmax
            # so to make mask additive we just set the 0s to -inf
            # assuming mask is originally 0s and 1s
            # 0s should be np inf, 1s should become 0s?
            additive_mask = torch.where(attn_mask==0, -torch.inf, 0)
            dot_product += additive_mask
        
        # apply softmax, dropout, and use value
        # ok apply softmax along key dim
        y = F.softmax(dot_product, dim=2)
        # now we apply dropout
        y = self.dropout(y)
        # now we multiply by values
        # y is still N, S, T
        # values are N, T, D
        y = torch.matmul(y, value)
        return y  

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads

        # TODO: Initialize the following layers and parameters to perform attention
        self.head_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # TODO : Compute multi-head attention

        #project query, key and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        #after projection, split the embedding across num_heads
        #eg - expected shape for value is (N, H, T, D/H)
        # ok so originaly query is N, T, D
        # we need to split across num heads = H
        # so first we can reshape to N, T, H, D/h so that feature dim split into H channels
        # then swap dims
        # and 

        d_head = D//H
        query = query.view(N, S, H, d_head).transpose(1, 2)
        key = key.view(N, T, H, d_head).transpose(1, 2)
        value = value.view(N, T, H, d_head).transpose(1, 2)

        #compute dot-product attention separately for each head. Don't forget the scaling value!
        #Expected shape of dot_product is (N, H, S, T)
        # query is N, H, S, D/H
        # value is N, H, T, D/H
        # scale by D/H this time
        dot_product = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_head)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            # Hint : If mask[i,j] = 0, we want softmax(QKT[i,j] + additive_mask[i,j]) to be 0
            # Think about what inputs make softmax 0.
            # same ideas as single head
            additive_mask = torch.where(attn_mask==0, -torch.inf, 0)
            dot_product += additive_mask
        
        # apply softmax, dropout, and use value
        # dot_product is (N, H, S, T)
        # we still want to softmax along key/value dim
        attn = F.softmax(dot_product, dim=3)
        attn = self.dropout(attn)

        # value is N, H, T, D/H
        y = torch.matmul(attn, value)

        # y has shape N, H, S, D/H
        # we want to get back to N, S, D
        # concat embeddings from different heads, and project
        # basically have swap then reshape reverse of before
        # need to use contiguous so that we can reshape without memory errors

        y = y.transpose(1,2).contiguous().view(N, S, D)
        output = self.head_proj(y)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # TODO - use torch.nn.Embedding to create the encoding. Initialize dropout layer.

        # 5000 x embed dim look up table for 
        self.encoding = nn.Embedding(num_embeddings=max_len, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
      
    def forward(self, x):
        N, S, D = x.shape
        # TODO - add the encoding to x
        # need to create N, S, D encoding
        # need 1, S indices for proper brodcasting
        indices = torch.arange(S, device=x.device).unsqueeze(0)

        output = x + self.encoding(indices)
        output = self.dropout(output)
   
        return output


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for self_attn.
        self.self_attn = MultiHeadAttentionLayer(embed_dim=input_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, mask):
        ############# TODO - Self-attention on the sequence, using the mask. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        attn_out = self.self_attn(seq, seq, seq, mask)
        out = self.dropout(attn_out)
        out += seq  # residual connection
        out = self.layernorm(out)
        return out

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # TODO: Initialize the following. Use MultiHeadAttentionLayer for cross_attn.
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, cond):
        ############# TODO - Cross-attention on the sequence, using conditioning. Add dropout to attention layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        attn_out = self.cross_attn(seq, cond, cond)  # no masking
        out = self.dropout(attn_out)
        out += seq  # residual connection
        out = self.norm(out)
        return out
class SwiGLU(nn.Module):
    #(xW1​+b1​)⊙Swish(xW2​+b2​)
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, input_dim)  #project back up
    def forward(self, x):
        x = self.linear1(x) * F.silu(self.linear2(x))
        return self.proj(x)
    
class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        # TODO: Initialize the following. 
        # MLP has the following layers : linear, relu, dropout, linear ; hidden dim of linear is given by dim_feedforward
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(dim_feedforward, input_dim)
        # )
        self.mlp = SwiGLU(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       

    def forward(self, seq):
         ############# TODO - MLP on the sequence. Add dropout to mlp layer output.
        # Then add a residual connection to the original input, and finally apply normalization. #############################
        out = self.mlp(seq)
        out = self.dropout(out)
        out += seq  # residual connection
        out = self.norm(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        # TODO - get caption and feature embeddings 
        # Don't forget position embeddings for captions!
        # expected caption embedding output shape : (N, T, D)

        # Unsqueeze feature embedding along dimension 1
        # expected feature embedding output shape : (N, 1, D) 

        # uhhh ok caption should be N, T, D
        # caption input is N, T
        caption_embedding = self.caption_embedding(captions)
        caption_embedding = self.positional_encoding(caption_embedding)

        ## ok now feature input is N, D
        feature_embedding = self.feature_embedding(features)
        feature_embedding = feature_embedding.unsqueeze(1)
        return feature_embedding, caption_embedding

    def get_causal_mask(self, _len):
        #TODO - get causal mask. This should be a matrix of shape (_len, _len). 
        # This mask is multiplicative
        # setting mask[i,j] = 0 means jth element of the sequence is not used 
        # to predict the ith element of the sequence.
        # indices where j > i should be 0
        mask = torch.tril(torch.ones((_len, _len), device=self.device))

        return mask
                                      
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions


