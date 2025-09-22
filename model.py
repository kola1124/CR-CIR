

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import os
import numpy as np
import operators
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
#from lib.utils import dummy_context_mgr
from clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from simple_tokenizer import SimpleTokenizer
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

def wasserstein_distance(p, q):
    """
    计算 Wasserstein 距离
    p 和 q 是形状为 (bz, 512) 的张量，表示多个一维概率分布
    """
    # 确保输入是归一化的概率分布
    p = F.normalize(p, dim=-1, p=2)
    q = F.normalize(q, dim=-1, p=2)
 
    
    # 计算累积分布函数（CDF）
    p_cdf = torch.cumsum(p, dim=1)  # 按行计算CDF
    q_cdf = torch.cumsum(q, dim=1)  # 按行计算CDF
    
   
    dist = torch.sum(torch.abs(p_cdf - q_cdf), dim=1)  
    return dist

def kl_divergence(p, q):
    """
    计算Kullback-Leibler散度
    p 和 q 是概率分布
    """
    # 归一化概率分布
    p = p / p.sum(dim=1, keepdim=True)  # 按行归一化
    q = q / q.sum(dim=1, keepdim=True)  # 按行归一化
    
    # 避免对数零的问题
    p = torch.clamp(p, min=1e-10)  # 防止零值
    q = torch.clamp(q, min=1e-10)  # 防止零值
    
    # 计算KL散度
    return (p * (p / q).log()).sum(dim=1)  # 按行计算

def js_divergence(p, q):
    """
    计算Jensen-Shannon散度
    p 和 q 是形状为 (bz, 1024) 的概率分布
    """
    # 归一化概率分布
    p = p / p.sum(dim=1, keepdim=True)  # 按行归一化
    q = q / q.sum(dim=1, keepdim=True)  # 按行归一化
    
    # 计算平均分布
    m = 0.5 * (p + q)

    # 计算Jensen-Shannon散度
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


class HingebasedCrossAttentionCLIP(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        # attention proj
        self.query_ref1 = nn.Linear(embed_dim,embed_dim)
        self.key_text1 = nn.Linear(embed_dim,embed_dim)
        # self.query_text1 = nn.Linear(embed_dim,embed_dim)
        self.key_tar1 = nn.Linear(embed_dim,embed_dim)
        self.value1 = nn.Linear(embed_dim,embed_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.query_ref2 = nn.Linear(embed_dim,embed_dim)
        self.key_text2 = nn.Linear(embed_dim,embed_dim)
        self.key_tar2 = nn.Linear(embed_dim,embed_dim)
        self.value2 = nn.Linear(embed_dim,embed_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU(inplace=True)


    def forward(self, reference_embeds, caption_embeds, target_embeds):
        psudo_T = self.hca_T_share_text(reference_embeds, caption_embeds, target_embeds)
        return psudo_T
    
    
    def hca_T_share_text(self, reference_embeds, caption_embeds, target_embeds):

        # bs, hi, h, w = reference_embeds.size()
        # #embeddings to tokens  bs x length x hidden    bs 81 2560
        # reference_embeds = reference_embeds.view(bs,h*w,hi)
        # target_embeds = target_embeds.view(bs,h*w,hi)
        #dim compact bs 81 640  linear降维
        # reference_embeds = self.relu1(self.fc1(reference_embeds))
        # target_embeds = self.relu1(self.fc1(target_embeds))
        
        #print('reference_embeds', reference_embeds.shape) #reference_embeds torch.Size([16, 81, 640])
        #print('target_embeds', target_embeds.shape) #target_embeds torch.Size([16, 81, 640])
        
        attA = self.multiply(self.query_ref1(reference_embeds), self.key_text1(caption_embeds)) / math.sqrt(640) #Ar2c
        #print('attA', attA.shape) #16 81 81
        attB = self.multiply(self.key_text1(caption_embeds), self.key_tar1(target_embeds)) / math.sqrt(640)
        #print('attB', attB.shape) #16 81 81
        attC = self.dropout1(F.softmax(torch.matmul(attA, attB), dim=-1))
        #print('attC', attC.shape) #attC torch.Size([16, 81, 81])
        psudo_T = torch.matmul(attC , self.value1(target_embeds))
        return psudo_T[:,0,:] #out torch.Size([16, 640])
    
    def multiply(self, embedsA, embedsB):
        #print('embeds_A', embedsA.shape) # 16 81 640
        #print('embeds_B', embedsB.shape) # 16 81 640
        bs, len_a , dim = embedsA.shape
        bs, len_b , dim = embedsB.shape

        # 扁平化
        embedsA = embedsA.view(bs, -1, dim)  # 形状为 bs x (length_a * dim)
        embedsB = embedsB.view(bs, -1, dim)  # 形状为 bs x (length_b * dim)
        #print('embedsA', embedsA.shape) #embedsA torch.Size([16, 81, 640])

        # 点积计算
        attention_scores_flat = torch.matmul(embedsA, embedsB.transpose(-1, -2))  # 转置 Key 的维度
        #print('attention_scores_flat', attention_scores_flat.shape) #attention_scores_flat torch.Size([16, 81, 81])
        # 还原形状
        attention_scores = attention_scores_flat.view(bs, len_a, len_b)

        return attention_scores


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    #print('============tokenizer',tokenizer.encoder.keys())
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    #print('tokens', tokens.shape)
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
            length = 77
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    else:
        length = len(tokens)
    result[:len(tokens)] = torch.tensor(tokens)
    return result, length


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x
       
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.maxpool1d = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, features, text, atten):
        #token_indices = text.argmax(dim=-1)  # 形状 (bs, )
        cls_token = features[torch.arange(features.shape[0]), text.argmax(dim=-1)]
        #print('cls_tokenss', cls_token.shape)
        #print('atten_no_tokens', atten_no_tokens.shape) #atten_no_tokens torch.Size([32, 76, 76])
        weighted_feature = torch.bmm(atten, features)
        # cls_token = self.fc(cls_token)
        #print('features', features.shape) #features torch.Size([32, 256, 1024])
        features = self.fc(weighted_feature)
        #features = self.mlp(weighted_feature)
        features = self.maxpool1d(weighted_feature.transpose(1, 2)).squeeze(-1)
        #print('features', features.shape)
        #print('cls_token', cls_token.shape)
        features = features + cls_token
        return F.normalize(features, p=2, dim=-1).float()
     
class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=1024,ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        #self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.maxpool1d = nn.AdaptiveAvgPool1d(output_size=1)
        
    def forward(self, base_features, atten):
        cls_token = base_features[:, 0, :] 
        # cls_token = self.fc(cls_token)
        weighted_feature = torch.bmm(atten, base_features) #bs 16 1024
        features = self.fc(weighted_feature)
        #print('features', features.shape) #features torch.Size([32, 256, 1024])
        #features = self.mlp(weighted_feature)
        features = self.maxpool1d(weighted_feature.transpose(1, 2)).squeeze(-1)
        
        features = features + cls_token
        
        return F.normalize(features, p=2, dim=-1).float()

def compute_mlm(scores, labels):
    
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        #print('xxxxx=========', x.shape)
        #print('arg_max') #用这个
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens

class build_text_encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = clip_model.text_pool_type
        self.register_buffer('attn_mask', clip_model.attn_mask, persistent=False)
        
    
    def forward(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        #print('len',len(self.transformer(x, attn_mask=self.attn_mask)))
        x, atten_weight = self.transformer(x, attn_mask=self.attn_mask)
        #print('atten_weight_text', atten_weight.shape) #atten_weight_text torch.Size([16, 77, 77])
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        #print('用这个') 会打印
        token, _ = text_global_pool(x, text, self.text_pool_type) #只取了token
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                #print('nn.Linear')
                token = self.text_projection(token)
            else:
                #print('else else') #用的这行
                token = token @ self.text_projection
                base_feature = x @ self.text_projection
        if normalize:
            token = F.normalize(token, dim=-1)
        return token, atten_weight, base_feature
        
class referenceAttImg(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.target_caption = nn.Linear(embed_dim, embed_dim)
        
        
    def forward(self, reference_caption, reference_image, target_caption):
        Q = self.query(reference_caption)
        K = self.key(reference_image)
        V = self.value(reference_image)
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        
        d_K = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(K)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        # Calculate the output as a weighted sum of the values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
    
    def att(self, Q, K, V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        
        d_K = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(K)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        # Calculate the output as a weighted sum of the values
        output = torch.matmul(attention_weights, V)

        return F.normalize(output, p=2, dim=-1)
        
        
       
    
    def inatt(self, Q, K, V):
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        
        d_K = K.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(K)
        
        attention_weights = 1 - F.softmax(scaled_attention_logits, dim=-1)
        
        # Calculate the output as a weighted sum of the values
        output = torch.matmul(attention_weights, V)

        return F.normalize(output, p=2, dim=-1)
          
class TextFusion(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        
        
    def forward(self, reference_caption, reference_image, modification_text):
        query = self.Q(modification_text)
        # key = self.K(reference_image)
        # value = self.K(reference_image)
        key = self.K(reference_caption)
        value = self.K(reference_caption)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) /torch.sqrt(torch.tensor(self.embed_dim))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (1, 3, 3)

        inverse_attn_weights = 1 - F.softmax(attn_scores, dim=-1)
        # 计算加权的 V，得到注意力输出
        related_target_output = torch.matmul(attn_weights, value)  
        unrelated_target_output = torch.matmul(inverse_attn_weights, value)
        
        output = reference_caption + related_target_output + modification_text
        return output, unrelated_target_output
        
class ImageFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        # self.V = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        #self.att = nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=True)
    def forward(self, reference_image, modification_txt):
        query = self.Q(modification_txt)
        key = self.K(reference_image)
        value = self.K(reference_image)
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) /torch.sqrt(torch.tensor(self.embed_dim))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (1, 3, 3)

        inverse_attn_weights = 1 - F.softmax(attn_scores, dim=-1)
        # 计算加权的 V，得到注意力输出
        related_target_output = torch.matmul(attn_weights, value)  
        unrelated_target_output = torch.matmul(inverse_attn_weights, value)
        input = related_target_output + reference_image + modification_txt
        #output_img = self.att(related_target_output, input, input)
    
        return input, unrelated_target_output
        

class DQU_CIR(nn.Module):
    def __init__(self, hidden_dim=512, dropout = 0.5, use_hog=False, mask_ratio=0., mlm=False, alpha=0.1):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='ICME-main/src/CLIP-ViT-B-16-laion2B-s34B-b88K /open_clip_pytorch_model.bin')
      
        self.proj = nn.Linear(1, 1) 
        self.hca = ImageFusion(512)
        self.tca = TextFusion(512)
        self.alpha = alpha
        clip_caption, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='ICME-main/src/CLIP-ViT-B-16-laion2B-s34B-b88K /open_clip_pytorch_model.bin')
        self.clip_encoder = build_text_encoder(clip_caption)
        self.clip = self.clip.float()
        self.mask_ratio = mask_ratio
        self.diffimg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
        )
        
        self.difftxt_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
        )
        #self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        #print('Using SimpleTokenizer...')
        # self.tokenizer = SimpleTokenizer()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        #print('use_hog', self.use_hog)
        self.use_hog = use_hog
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        
   
        self.mlm = mlm
        self.embed_dim = hidden_dim
        
        if self.mlm:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=4,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, 49408))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
            
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = [x, None]
        x = self.cross_modal_transformer(x)[0]
        
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x        
    
    def extract_img_fea(self, x):
        #vision_cfg CLIPVisionCfg(layers=32, width=1280, head_width=80, mlp_ratio=4.0, patch_size=14, image_size=224, ls_init_value=None, patch_dropout=0.0, attentional_pool=False, attn_pooler_queries=256, attn_pooler_heads=8, no_ln_pre=False, pos_embed_type='learnable', final_ln_after_pool=False, pool_type='tok', output_tokens=False, act_kwargs=None, norm_kwargs=None, timm_model_name=None, timm_model_pretrained=False, timm_pool='avg', timm_proj='linear', timm_proj_bias=False, timm_drop=0.0, timm_drop_path=None)
        #Loading pretrained ViT-H-14 weights (./laion2B-s32B-b79K/open_clip_pytorch_model.bin).
        image_features, atten_image, base_feature = self.clip.encode_image(x)
        #print('image_feature.shape===', base_feature.shape) #image_feature.shape=== torch.Size([16, 197, 512])
        #print('image_features', image_features.shape) #image_features torch.Size([16, 1024])
        #print('atten_image', atten_image.shape) #atten_image torch.Size([16, 257, 257])
        
        #print('base_img', base_feature.shape) #base_img torch.Size([16, 257, 1024])
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features, atten_image, base_feature
    
    def extract_text_fea(self, txt):
        #print('txt====', txt)
        txt = self.tokenizer(txt).cuda()
        #print('txt shape====', txt.shape)
        #print('txt', txt.shape) #
        text_features, atten_text, base_feature = self.clip.encode_text(txt)
        #print('text===>', base_feature.shape) #text===> torch.Size([16, 77, 512])
        #print('atten_text', atten_text.shape) #atten_text torch.Size([16, 77, 77])
        #print('text_features', text_features.shape) #text_features torch.Size([16, 1024])
        #print('base_txt', base_feature.shape) #base_txt torch.Size([16, 77, 1024])
        text_features = F.normalize(text_features, p=2 ,dim=-1)
        return text_features, atten_text, base_feature
    # def extract_fusion(self, textual_query, hog_features):
    #     textual_query_raw, atten_text, base_text = self.extract_text_fea(textual_query)
    def extract_image_feats(self, textual_query, visual_query):
        textual_query_raw, atten_text, base_text = self.extract_text_fea(textual_query)  # 16 257 512 VITB16是512
        visual_query_raw, atten_image, base_image = self.extract_img_fea(visual_query) #16 77 512
        
        B, N, D = base_image.shape
        text_feature = textual_query_raw.unsqueeze(1).expand(-1, N, -1)
        image_feats = base_image+text_feature
        return F.normalize(image_feats, p=2, dim=-1)
        
    def extract_query(self, textual_query, visual_query, caption_query, only_image=False):
        if caption_query:
        #txt = self.tokenizer(textual_query).cuda()
        
            textual_query_raw, atten_text, base_text = self.extract_text_fea(textual_query)  # 16 257 512 VITB16是512
            visual_query_raw, atten_image, base_image = self.extract_img_fea(visual_query) #16 77 512
            #caption_query_raw, _,  _ = self.extract_text_fea(textual_query)
            
            caption_query_raw, _,  _ = self.clip_encoder(self.tokenizer(caption_query).cuda())
            #visual_query_raw, atten_image, base_image = self.extract_img_fea(visual_query) #16 77 512
            #print('visual_query', visual_query_raw.shape)
            # print('textual_query_raw', textual_query_raw.shape)
            textual_query_raw = F.normalize(textual_query_raw, p=2, dim=-1)
            visual_query_raw = F.normalize(visual_query_raw, p=2, dim=-1)
            caption_query_raw = F.normalize(caption_query_raw, p=2, dim=-1)
            
            
            
       
            if only_image:
                return F.normalize(visual_query_raw+textual_query_raw, p=2, dim=-1), F.normalize(caption_query_raw, p=2,dim=-1), visual_query_raw
            else:
                visual, _ = self.hca(visual_query_raw, textual_query_raw)
                txtual, _ = self.tca(caption_query_raw, visual_query_raw, textual_query_raw)
                return F.normalize(visual, p=2, dim=-1), F.normalize(txtual, p=2,dim=-1)
            
        else:
            textual_query_raw, atten_text, base_text = self.extract_text_fea(textual_query)  # 16 257 512 VITB16是512
            visual_query_raw, atten_image, base_image = self.extract_img_fea(visual_query) #16 77 512
            
            textual_query_raw = F.normalize(textual_query_raw, p=2, dim=-1)
            visual_query_raw = F.normalize(visual_query_raw, p=2, dim=-1)
            
            query_raw, un_query = self.hca(visual_query_raw, textual_query_raw)
            #query_raw = visual_query_raw + textual_query_raw 
        # query_raw = dynamic_scaler_raw * textual_query_raw + (1 - dynamic_scaler_raw) * visual_query_raw
            return F.normalize(query_raw, p=2, dim=-1)
     
    def extract_target(self, target_img, caption):
        if caption:
            target_img_fea, atten_image, base_image = self.extract_img_fea(target_img)
            caption_txt_fea, _, _ = self.clip_encoder(self.tokenizer(caption).cuda())
            #caption_txt_fea, _, _ = self.extract_text_fea(caption)
            
            target_img_fea = F.normalize(target_img_fea, p=2, dim=-1)
            caption_txt_fea = F.normalize(caption_txt_fea, p=2, dim=-2)
            #print('target_img_fea', target_img_fea.shape)
            #target_img_tse = self.extract_img_tse(target_img)
            #print('target_img_fea', target_img_fea.shape)
            return F.normalize(target_img_fea, p=2, dim=-1), F.normalize(caption_txt_fea, p=2, dim=-1)
            
        else:
            target_img_fea, atten_image, base_image = self.extract_img_fea(target_img)
        #target_img_fea = self.combiner_fc(torch.cat((caption_txt_fea, target_img_fea), dim=-1))
        #target_img_fea = target_img_fea
            return F.normalize(target_img_fea, p=2, dim=-1)
      
    
    
    def compute_loss(self, textual_query, visual_query, target_img, batch):

        loss = {}
        if self.mlm:
            mlm_ids = batch['mlm_ids'].cuda()
            #print('mlm_ids', mlm_ids.shape)
            
            _, _, mlm_feats = self.clip.encode_text(mlm_ids)
            image_feats = self.extract_image_feats(batch['captions'], batch['candidate'])
            #print('mlm_feats', mlm_feats.shape, 'image_feats', image_feats.shape)
            
            x = self.cross_former(mlm_feats.cuda(), image_feats, image_feats)
            x = self.mlm_head(x)
            #print("帮帮孩子啊, 帮你吗")
            #print('xxxxxxxxx', x.shape) xxxxxxxxx torch.Size([16, 77, 49408])
            # parser.add_argument("--vocab_size", type=int, default=49408)
            scores = x.float().reshape(-1, 49408)
            mlm_labels = batch['mlm_labels'].reshape(-1).cuda()
            mlm_loss = compute_mlm(scores, mlm_labels)
            loss['hog_loss'] = mlm_loss
        #print('self.use_hog===', self.use_hog)
        # if self.use_hog:
        #     output_mask = mask_chosed.to(bool)
        #     hog_preds = self.projections(outputs_hog[:, 1:, :]) #因为区域并不对应，所以indices并不准确，不能这么监督
        #     #print('hog_predssssss', hog_preds.shape) #torch.Size([16, 256, 108])
        #     #hog_preds = hog_preds[output_mask]
        #     #print('hog_preds', hog_preds.shape) #torch.Size([1216, 108])
        #     #print('target_img-----', target_img.shape) # 16 3 224 224
        #     #print('output_mask-----', output_mask.shape) # 16 256
        #     hog_labels = self._get_hog_label_2d(target_img, output_mask, block_size=14)
        #     #print('hog_labels---------', hog_labels.shape) #torch.Size([1216, 108])
        #     hog_loss = self.mse_func(hog_preds, hog_labels) 
        #     loss['hog_loss'] = hog_loss

        #query_feature = self.extract_query(textual_query, visual_query) 
        #print('--------', batch.keys())
        #query_feature, query_txt_feature, query_visual_feature = self.extract_query(textual_query, visual_query, None)
        query_feature, txtual  = self.extract_query(textual_query, visual_query, batch['candidate_caption'])
        #textual_query_raw, atten_text, base_text = self.extract_text_fea(textual_query)
        #query_feature = F.normalize((query_feature+textual_query_raw), p=2, dim=-1)
        #target_feature = self.extract_target(target_img, batch['target_caption'])
        target_feature, txttar = self.extract_target(target_img, batch['generated_target_caption'])
       
        batch_size = query_feature.shape[0]
        
        # loss['nce_loss'] = self.ranking_nce_loss(query_feature, target_feature)    
        # loss['nce_loss_tse'] = self.ranking_nce_loss(query_feature_tse, target_feature_tse)   
        
        #loss['sdm_loss'] = self.ranking_sdm_loss(query_feature, target_feature)
        #loss['sdm_loss_tse'] = self.ranking_sdm_loss(query_feature_tse, target_feature_tse)
        # loss['nce_loss'] = self.ranking_nce_loss(query_feature, target_feature)
        # loss['nce_loss_tse'] = self.ranking_nce_loss(query_feature_tse, target_feature_tse)
        
        nce_loss = self.ranking_nce_loss(query_feature, target_feature)
        hog_loss = self.ranking_nce_loss(txtual, txttar)
        kl_loss = self.similarity_equal(query_feature, target_feature, txtual, txttar, self.alpha) #这个有效

        loss['nce_loss'] = nce_loss
        loss['hog_loss'] = hog_loss
        #loss['kl_loss'] = kl_loss
        loss['kl_loss'] = kl_loss * 0.5  #hyper是0.5是会上点的shoes    0.8对fashioniq上点
        #loss['hog_loss'] = self.ranking_nce_loss(query_feature, target_text_feature)
        #loss['nce_loss_tse'] = (nce_loss_tse * label_hat.cuda()).sum()
        return loss


    def js_loss_vectorized(self, tensor1, tensor2):
        """
        计算两个张量在第一维度上的Jensen-Shannon损失
        
        参数:
        tensor1 (torch.Tensor): 第一个概率分布张量，形状为 (16, 512)
        tensor2 (torch.Tensor): 第二个概率分布张量，形状为 (16, 512)
        
        返回:
        js_losses (torch.Tensor): Jensen-Shannon损失值，形状为 (16,)
        """
        # 应用softmax确保每行是一个概率分布
        p = torch.softmax(tensor1, dim=1)  # 计算每行的softmax
        q = torch.softmax(tensor2, dim=1)  # 计算每行的softmax
        
        # 计算中间分布M
        m = 0.5 * (p + q)
        
        # 避免计算log(0)
        p = p + 1e-10
        q = q + 1e-10
        m = m + 1e-10
        
        # 计算Kullback-Leibler散度
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)  # 沿着特征维度求和
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)  # 沿着特征维度求和
        
        # 计算Jensen-Shannon散度
        js_losses = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_losses.sum()
    
    def ranking_nce_loss(self, query, target, sims=False):
        x = torch.mm(query, target.t())
        #x_t = x.t()
        
        #print('x.shape', x.shape) torch.Size([16, 16])
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight * x, labels)
        #loss_t = F.cross_entropy(self.loss_weight * x_t, labels)
        #print('loss', loss.shape)
        # if sims:
        #     return loss, x.diag()
        
        return loss
    
    def ranking_rce_loss(self, query, target, tau=0.1):
        eps = 1e-7
        scores = torch.mm(query, target.t())
        mask = torch.eye(scores.shape[0]) + eps
        mask = mask.cuda()
        
        scores = (scores / tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True))
        t2i = scores.t() / (scores.t().sum(1, keepdim=True))
        
        cost_i2t_r = - (mask.log()*i2t).sum(1).mean()
        cost_t2i_r = - (mask.log()*t2i).sum(1).mean()
        cost_i2t = -i2t.diag().log().mean()
        cost_t2i = -t2i.diag().log().mean()
        return 0.5*(cost_i2t_r + cost_t2i_r + cost_i2t + cost_t2i)
    
    def compute_Rce_per(self, scores, logit_scale):
        eps = 1e-7
        # scores = torch.mm(query, target.t())
        mask = torch.eye(scores.shape[0]) + eps
        mask = mask.cuda()
        
        scores = (scores / 0.1).exp()
        i2t = scores / (scores.sum(1, keepdim=True))
        t2i = scores.t() / (scores.t().sum(1, keepdim=True))
        
        cost_i2t_r = - (mask.log()*i2t)
        cost_t2i_r = - (mask.log()*t2i)
        cost_i2t = -i2t.diag().log()
        cost_t2i = -t2i.diag().log()
        return 0.5*(cost_i2t_r + cost_t2i_r + cost_i2t + cost_t2i), 
    
    def compute_InfoNCE_per(self, scores, logit_scale):
    
        # cosine similarity as logits
        logits_per_image = logit_scale * scores
        logits_per_text = logits_per_image.t()

        p1 = F.softmax(logits_per_image, dim=1)
        p2 = F.softmax(logits_per_text, dim=1)

        loss = (- p1.diag().log() - p2.diag().log())/2    
        return loss, scores.diag()
    
    def ranking_sdm_loss(self, query, target, logit_scale = 50, epsilon=1e-8):
        x = torch.mm(query, target.t())
        batch_size = x.shape[0]
        pid = torch.tensor(range(x.shape[0])).long().cuda()
        #print('pid', pid.shape)
        #print('batch_size', batch_size, type(batch_size))
        pid = pid.reshape((batch_size, 1))
        pid_dist = pid - pid.t() #这里做的是差值
        labels = (pid_dist == 0).float()
        
        t2i_cosine_theta = x
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

        return loss.sum()/batch_size
        # labels = torch.autograd.Variable(labels).cuda()

    def negative_matching_loss(self, query, target, logit_scale=50, epsilon=1e-8):
    # 计算相似度矩阵
        x = torch.mm(query, target.t())  # (batch_size, batch_size)
        batch_size = x.shape[0]
        
        # 生成标签，匹配样本对标签为 1，非匹配样本对标签为 0
        pid = torch.arange(batch_size).long().to(query.device)
        pid = pid.view(batch_size, 1)
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()  # 匹配样本对为 1，非匹配样本对为 0
        
        # 对匹配样本的相似度施加负对数惩罚
        matching_similarity = labels * x  # 匹配样本对的相似度矩阵
        
        # 最大化匹配相似度的负对数损失
        loss = -torch.log(1 + matching_similarity)  # 使用 log(1 + sim) 保证正值
        loss = loss.sum()  # 对所有匹配样本求和
        
        return loss / batch_size

    def logarithmic_similarity_loss(self, query, target, epsilon=1e-8):
    # 计算相似度矩阵
        x = torch.mm(query, target.t())  # (batch_size, batch_size)
        batch_size = x.shape[0]
        
        # 生成标签
        pid = torch.arange(batch_size).long().to(query.device)
        pid = pid.view(batch_size, 1)
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()  # 匹配样本对为 1，非匹配样本对为 0
        
        # 提取匹配样本对的相似度
        matching_similarity = labels * x  # 匹配样本对的相似度矩阵
        
        # 使用 log(1 + similarity + epsilon)，避免负值
        loss = torch.log(1 + matching_similarity + epsilon)  # epsilon 防止数值问题
        
        return loss.sum() / batch_size
    
    def exp_log_similarity_loss(self, query, target):
    # 计算相似度矩阵
        x = torch.mm(query, target.t())  # (batch_size, batch_size)
        batch_size = x.shape[0]
        
        # 生成标签
        pid = torch.arange(batch_size).long().to(query.device)
        pid = pid.view(batch_size, 1)
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()  # 匹配样本对为 1，非匹配样本对为 0
        
        # 提取匹配样本对的相似度
        matching_similarity = labels * x  # 匹配样本对的相似度矩阵
        
        # 使用 log(1 + exp(similarity)) 确保正值
        loss = torch.log(1 + torch.exp(matching_similarity))
        
        return loss.sum() / batch_size
    # def similarity_equal(self, query1, target1, query2, target2, alpha=0.1):
    #     cosine_similarity1 = F.cosine_similarity(query1, target1, dim=1)
    #     cosine_similarity2 = F.cosine_similarity(query2, target2, dim=1)
    #     #print('cosine_similarity1', cosine_similarity1.shape)
    #     #mse = ((cosine_similarity1 - cosine_similarity2)**2)
    #     #mse = torch.mean((cosine_similarity1 - cosine_similarity2) ** 2)
    #     #print('msemse', mse.shape)
    #     # 或者使用 L2 范数来表示差的均方误差
    #     mse = nn.MSELoss()
    #     l2_norm = torch.norm(cosine_similarity1 - cosine_similarity2, p=2)  # 计算 L2 范数
    #     #mse_from_norm = l2_norm ** 2 / cosine_similarity1.size(0)  # 平方后取平均
        
    #     return F.relu(l2_norm - alpha)
    
    def similarity_equal(self, query1, target1, query2, target2, alpha=0.1):
        cosine_similarity1 = F.cosine_similarity(query1, target1, dim=1)
        cosine_similarity2 = F.cosine_similarity(query2, target2, dim=1)
        #print('cosine_similarity1', cosine_similarity1.shape)
        #mse = ((cosine_similarity1 - cosine_similarity2)**2)
        #mse = torch.mean((cosine_similarity1 - cosine_similarity2) ** 2)
        #print('msemse', mse.shape)
        # 或者使用 L2 范数来表示差的均方误差
        mse = nn.MSELoss()
        l2_norm = torch.norm(cosine_similarity1 - cosine_similarity2, p=2)  # 计算 L2 范数
        #mse_from_norm = l2_norm ** 2 / cosine_similarity1.size(0)  # 平方后取平均
        
        return F.relu(l2_norm - alpha)
        
        
    def js_loss(self, p, q):
        """
        计算Jensen-Shannon损失
        
        参数:
        p (torch.Tensor): 第一个概率分布，形状为 (n,)
        q (torch.Tensor): 第二个概率分布，形状为 (n,)
        
        返回:
        js_loss (torch.Tensor): Jensen-Shannon损失值
        """
        # 避免计算log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # 计算中间分布M
        m = 0.5 * (p + q)
        
        # 计算Kullback-Leibler散度
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum()
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum()
        
        # 计算Jensen-Shannon散度
        js = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js
    
    def kl_divergence(self, p, q):
        """
        计算两个一维向量的Kullback-Leibler散度
        
        参数:
        p (torch.Tensor): 第一个一维概率分布，形状为 (n,)
        q (torch.Tensor): 第二个一维概率分布，形状为 (n,)
        
        返回:
        kl_div (torch.Tensor): Kullback-Leibler散度值
        """
        # 避免计算log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # 计算Kullback-Leibler散度
        kl_div = (p * (torch.log(p) - torch.log(q))).sum()
        
        return kl_div
 
class VIT32(nn.Module):
    def __init__(self, hidden_dim=1024, dropout = 0.5, use_hog=False, mask_ratio=0.):
        super().__init__()
        # self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join('../models/laionCLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin'))
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B/32', pretrained='./laion2B-s32B-b79K/open_clip_pytorch_model.bin')
        # print('self.clip', self.clip)
        # print()
        self.clip = self.clip.float()
        self.mask_ratio = mask_ratio
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.use_hog = use_hog
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        
        self.combiner_fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                         nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.scaler_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())
        # self.visual_emb_layer = VisualEmbeddingLayer()
        # self.textual_emb_layer = TexualEmbeddingLayer()
        self.head_dim = 1280
        self.mask_token_dim = (1, 1, self.head_dim)
        self.mask_token = nn.Parameter(torch.zeros(*self.mask_token_dim), requires_grad=True).cuda()
        #self.projections = nn.Linear(hidden_dim, num_class)
        if self.use_hog:
            nbins = 9
            cell_sz = 7
            self.hogs = operators.HOGLayerC(
                nbins=nbins,
                pool=cell_sz
            )
            
            self.hogs.cuda()
            num_class = int(nbins*3*(14/cell_sz)*(14/cell_sz)) 
            #print('num_class----', num_class) # 108
            
            self.projections = nn.Linear(hidden_dim, num_class, bias=True) # 768-
            if isinstance(self.projections, nn.Linear):
                nn.init.trunc_normal_(self.projections.weight, std=0.02)
                if isinstance(self.projections, nn.Linear) and self.projections.bias is not None:
                    nn.init.constant_(self.projections.bias, 0)  
            self.projections.cuda()
            self.mse_func = nn.MSELoss(reduction="mean")
            
            
            self.ln_pre_t = LayerNorm(hidden_dim)
            self.ln_pre_i = LayerNorm(hidden_dim)
            self.ln_post = LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim,
                                                    hidden_dim // 64,
                                                    batch_first=True)
            
            self.cross_modal_transformer = Transformer(width=hidden_dim,
                                                       layers = 3,
                                                       heads = hidden_dim // 64)
            
            scale = self.cross_modal_transformer.width**-0.5
            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        else:
            self.hogs = None
            self.projections = None
 
class CLIP_Mapper(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_Mapper, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        # for param in model.parameters():
        #     param.requires_grad = False
            
        self.proj1280 = nn.Parameter(torch.randn(1024, 1280))
    def define_module(self, model):
        #做出跟clip_model一样的模型结构， 这些结构编辑好的model.conv1里面都有
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.proj = model.proj
    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor, prompts: torch.Tensor):
        x = img.type(self.dtype)
        #print('img', x.shape)
        #print('xxxxx',x.shape) #xxxxx torch.Size([16, 1024, 16, 16])
        prompts = prompts.type(self.dtype) @ self.proj1280
        #print('prompts', prompts.shape) # 16 8 1024
        grid = x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1) @ self.proj1280 # shape = [*, grid ** 2, width]
        #print('xxx=', x.shape) #torch.Size([16, 49, 1280])
        
        #print('class', self.class_embedding.shape) #1280
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        # shape = [*, grid ** 2 + 1, width]
        #print('xxx', x.shape) #xxx torch.Size([16, 50, 1024])
        #print('self.positional_embedding', self.positional_embedding.shape) #self.positional_embedding torch.Size([257, 1280])
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        #print('xxx', type(x)) tensor
        # Local features
        selected = [1,2,3,4,5,6,7,8]
        begin, end = 0, 12
        prompt_idx = 0
        #print('sss', type(x))
        for i in range(begin, end):
            if i in selected:
                #print('i if if', i)
                prompt = prompts[:,prompt_idx,:].unsqueeze(0) #prompts指的是外界的输入
                prompt_idx = prompt_idx+1
                #print('xxxxxx', type(x), len(x))
                #print('zheng que', type(prompt), type(x))
                x = torch.cat((x,prompt), dim=0)
                #print('xxx', type(x))
                x, atten_weight = self.transformer.resblocks[i](x)
                #print('x, x', x.shape)
                x = x[:-1,:,:]
                
            else:
                #print('i else else', i)
                # print('xxxxx', x.shape)
                x, atten_weight = self.transformer.resblocks[i](x) 
                #print('len x', x.shape)
        x = x @ self.proj
        
        return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 1024, grid, grid).contiguous().type(img.dtype)

class CLIP_Adapter(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, cond_dim, k, s, p, map_num, CLIP):
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block(in_ch, mid_ch, out_ch, cond_dim, k, s, p))
        #print('map_num===>', map_num) #map_num===> 4
        for i in range(map_num-1):
            self.FBlocks.append(M_Block(out_ch, mid_ch, out_ch, cond_dim, k, s, p))
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        self.CLIP_ViT = CLIP_Mapper(CLIP)
        self.conv = nn.Conv2d(1024, G_ch, 5, 1, 2)
        #
        self.fc_prompt = nn.Linear(cond_dim, CLIP_ch*8)

    def forward(self,out,c):
        prompts = self.fc_prompt(c).view(c.size(0),-1, self.CLIP_ch)
        #print('prompts===>', prompts.shape) #torch.Size([16, 8, 1024]) #text_prompts
        for FBlock in self.FBlocks:
            out = FBlock(out,c)
        #print('out.shape===>', out.shape) #out.shape===> torch.Size([16, 64, 16, 16])
        fuse_feat = self.conv_fuse(out)
        #print('fuse_feat', fuse_feat.shape) #fuse_feat torch.Size([16, 1024, 16, 16])
        map_feat = self.CLIP_ViT(fuse_feat,prompts)
        #print('map_feat', map_feat.shape) #map_feat torch.Size([16, 1024, 16, 16])
        return self.conv(fuse_feat+0.1*map_feat)
 
def get_G_in_out_chs(nf, imsize):
    
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size, mixed_precision, CLIP):
        #print('nfg==', ngf, 'nz==',nz, 'cond_dim==', cond_dim, 'imsize==', imsize, 'ch_size==', ch_size)
        #nfg== 64 nz== 100 cond_dim== 512 imsize== 256 ch_size== 3
        super(NetG, self).__init__()
        self.ngf = ngf
        self.mixed_precision = mixed_precision
        # build CLIP Mapper
        self.code_sz, self.code_ch, self.mid_ch = 16, 64, 32
        self.CLIP_ch = 1024
        self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch) #100 --> 7*7*64
        
        self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, self.code_ch, ngf*8, self.CLIP_ch, cond_dim+nz, 3, 1, 1, 4, CLIP)
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = 224
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, imsize))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, ch_size, 3, 1, 1),
            #nn.Tanh(),
            )

    def forward(self, image, text, eval=False): # x=noise, c=ent_emb #这里的C就是语言描述
        
        cond = torch.cat((image, text), dim=1)
        #print('cond===>', cond.shape) #torch.Size([16, 2048])
        out = self.mapping(self.fc_code(image).view(image.size(0), self.code_ch, self.code_sz, self.code_sz), cond)
        #print('out===>', out.shape) #
        for GBlock in self.GBlocks:
            out = GBlock(out, cond)
            
            
        out = self.to_rgb(out)
        #print('out.shape', out.shape) # 16 3 224 224
        return out
       
class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)  

class M_Block(nn.Module): #有点像融合模块
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        #print('hj', h.shape) #hj torch.Size([64, 32, 7, 7])
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        #print('hhhhh=====',h.shape) # 64 64 7 7
        return h

    def forward(self, h, c):
        #第一个输入是纯noisy，一个需要编辑的
        #一个是text的prompt和noisy的concat
        # print('h===',h.shape)
        # print('c===', c.shape)
        # h=== torch.Size([64, 64, 7, 7])
        # c=== torch.Size([64, 612])
        return self.shortcut(h) + self.residual(h, c)

class DFBLK(nn.Module): #有点像融合模块
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        #第一个输入是纯noisy，一个需要编辑的
        #一个是text的prompt和noisy的concat
        # print('x===',x.shape)
        # print('y===',y.shape)
        # x=== torch.Size([64, 32, 7, 7])
        # y=== torch.Size([64, 612])
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h
    
class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        
        #第一个输入是纯noisy，一个需要编辑的
        #一个是text的prompt和noisy的concat
        # print('x===',x.shape)
        # print('y===',y.shape)
        # x=== torch.Size([64, 32, 7, 7])
        # y=== torch.Size([64, 612])
        
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        #肯定都不是1，所以可以用
        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

class NetD(nn.Module):
# 定义鉴别器网络Dclass NetD(nn.Module):
    def __init__(self, ndf, imsize, ch_size, mixed_precision):
        super(NetD, self).__init__()
        self.mixed_precision = mixed_precision
        self.DBlocks = nn.ModuleList([
            #shortcut是true
            D_Block(1024, 1024, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(1024, 1024, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        self.main = D_Block(1024, 512, 3, 1, 1, res=True, CLIP_feat=False)

    def forward(self, h):
        #with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
        out = h[:,0] #只取第一个
            
        for idx in range(len(self.DBlocks)):
            out = self.DBlocks[idx](out, h[:,idx+1])
                
        out = self.main(out)
        
        return out
    
class D_Block(nn.Module): #这个就是Discriminator中的
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        # 64 3 768 7 7
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        # print('xxxxx', x.shape) #xxxxx torch.Size([64, 768, 7, 7])
        res = self.conv_r(x)
        # print('res', res.shape) #res torch.Size([64, 768, 7, 7])
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x



if __name__ =='__main__':
    # clip_caption, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/home/mail/2022s2/s230201705/conference_new/DQU-CIR-main/src/CLIP-ViT-B-16-laion2B-s34B-b88K /open_clip_pytorch_model.bin')
    # text_encoder = build_text_encoder(clip_caption)
    # tokenizer = open_clip.get_tokenizer('ViT-B-16')
    # captions = 'is solid black with no sleeves and is black with straps'
    # text = tokenizer(captions)
    # text_feature, _, base_text_faeture = text_encoder(text)
    # print('text_feature', text_feature.shape, base_text_faeture.shape)
    query1 = torch.randn([16])
    target1 = torch.randn([16])
    
    # query2 = torch.randn([16, 512])
    # target2 = torch.randn([16, 512])
    
    
    def js_loss(p, q):
        """
        计算两个一维向量的Jensen-Shannon损失
        
        参数:
        p (torch.Tensor): 第一个一维概率分布，形状为 (n,)
        q (torch.Tensor): 第二个一维概率分布，形状为 (n,)
        
        返回:
        js_loss (torch.Tensor): Jensen-Shannon损失值
        """
        # 避免计算log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # 计算中间分布M
        m = 0.5 * (p + q)
        
        # 计算Kullback-Leibler散度
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum()
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum()
        print('kl_pm', kl_pm)
        # 计算Jensen-Shannon散度
        js = 0.5 * kl_pm + 0.5 * kl_qm
        #js = F.relu()
        return F.relu(js)
            
    loss = js_loss(query1, target1)
    print(loss)
        