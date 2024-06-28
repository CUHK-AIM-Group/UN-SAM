import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from SAM.modeling.mask_decoder import DQDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d
from SAM.modeling.image_encoder import DTEncoder
from functools import partial
from typing import List, Tuple, Type


class SPGen(nn.Module):
    def __init__(self):
        super(SPGen, self).__init__()

        self.up1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2))

        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2))
   

        self.final1 = nn.Conv2d(256, 1, kernel_size=1)
        self.final2 = nn.Conv2d(256, 1, kernel_size=1)
        self.final3 = nn.Conv2d(64, 1, kernel_size=1)
        self.final4 = nn.Conv2d(16, 1, kernel_size=1)
        

    def forward(self, x):

        x1 = self.up1(x)
        x3 = self.up3(x)
        x4 = self.up4(x)

        x1 = self.final1(x1)
        x2 = self.final2(x)
        x3 = self.final3(x3)
        x4 = self.final4(x4)

        return x1, x2, x3, x4


class UNSAM(nn.Module):
    def __init__(self, domain_num):
        super(UNSAM, self).__init__()

        self.image_encoder = DTEncoder(
                depth=12,
                embed_dim=768,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11],
                window_size=14,
                out_chans=256,
                domain_num=domain_num
            )

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64), # 1024 // 16
            input_image_size=(1024, 1024),
            mask_in_chans=16,
            )
        
        self.mask_decoder = DQDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            class_num=domain_num + 1
            )


        self.spgen = SPGen()
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.num_mask_tokens = domain_num + 1
        self.mask_query = nn.Embedding(self.num_mask_tokens, 256)
    
    def forward(self, x, domain_seq, img_id=None):

        b = x.shape[0]
        image_embeddings = self.image_encoder(x, domain_seq)

        spgen1, spgen2, spgen3, spgen4 = self.spgen(image_embeddings)

        spgen1 = self.up1(spgen1)
        spgen2 = self.up2(spgen2)
        spgen3 = self.up3(spgen3)

        output_coarse = spgen4 + spgen3 + spgen2 + spgen1

        output_prob = torch.sigmoid(output_coarse.detach())
        output_prob[output_prob >= 0.95] = 1
        output_prob[output_prob < 0.95] = 0


        outputs_mask = []

        
        for idx in range(b): # for each batch 

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=output_prob[idx].unsqueeze(0),
            )

            image_embeddings_dec = image_embeddings[idx].unsqueeze(0)
            image_pe = self.prompt_encoder.get_dense_pe()

            # Mask 
            mask_tokens = self.mask_query.weight
            mask_tokens = mask_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
            tokens_mask = torch.cat((mask_tokens, sparse_embeddings), dim=1) # 1 x 5 x 256

            # Expand per-image data in batch direction to be per-mask
            mask_src = torch.repeat_interleave(image_embeddings_dec, tokens_mask.shape[0], dim=0)
            mask_src = mask_src + dense_embeddings # 1 x 256 x 64 x 64
            mask_pos_src = torch.repeat_interleave(image_pe, tokens_mask.shape[0], dim=0)  # 1 x 256 x 64 x 64

            low_res_masks = self.mask_decoder(
                src=mask_src,
                pos_src=mask_pos_src,
                tokens=tokens_mask,
                domain_seq=domain_seq
            )

            masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)

            outputs_mask.append(masks.squeeze(0))
            

        return torch.stack(outputs_mask, dim=0), output_coarse
