import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from SAM.modeling.mask_decoder import DQDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d
from typing import List, Tuple, Type


class SPGen(nn.Module):
    def __init__(self):
        super(SPGen, self).__init__()

        self.up1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1))

        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1))


        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1))

 
        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1))
   


        

    def forward(self, x):
        x1, x2, x3, x4 = x

        x1 = self.up1(x1)
        x2 = self.up1(x2)
        x3 = self.up1(x3)
        x4 = self.up1(x4)

        return x1, x2, x3, x4


class UNSAM(nn.Module):
    def __init__(self, image_encoder):
        super(UNSAM, self).__init__()

        self.image_encoder = image_encoder

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
            )


        self.spgen = SPGen()
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.num_mask_tokens = 12
        self.mask_query = nn.Embedding(self.num_mask_tokens, 256)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
        

    
    def forward(self, x, domain_seq, img_id=None):

        b = x.shape[0]
        image_embeddings, stacked_embedding = self.image_encoder(x, domain_seq)

        spgen_list = []

        for depth_num in range(7, 32, 8):
            spgen_list.append(stacked_embedding[:,depth_num,:,:].unsqueeze(0))

        spgen1, spgen2, spgen3, spgen4 = self.spgen(spgen_list)

        spgen1 = self.up1(spgen1)
        spgen2 = self.up2(spgen2)
        spgen3 = self.up3(spgen3)

        spgen1_prob = torch.sigmoid(spgen1.detach())
        spgen1_prob[spgen1_prob >= 0.95] = 1
        spgen1_prob[spgen1_prob < 0.95] = 0


        spgen2_prob = torch.sigmoid(spgen2.detach())
        spgen2_prob[spgen2_prob >= 0.95] = 1
        spgen2_prob[spgen2_prob < 0.95] = 0


        spgen3_prob = torch.sigmoid(spgen3.detach())
        spgen3_prob[spgen3_prob >= 0.95] = 1
        spgen3_prob[spgen3_prob < 0.95] = 0
        

        spgen4_prob = torch.sigmoid(spgen4.detach())
        spgen4_prob[spgen4_prob >= 0.95] = 1
        spgen4_prob[spgen4_prob < 0.95] = 0

        output_coarse = spgen1_prob + spgen2_prob + spgen3_prob + spgen4_prob
        output_coarse[output_coarse > 1] = 1

        outputs_mask = []

        
        for idx in range(b): # for each batch 

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=output_coarse[idx].unsqueeze(0),
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
            )

            masks = self.postprocess_masks(low_res_masks.squeeze(0), (256,256), (1024,1024))

            outputs_mask.append(masks)
            

        return torch.stack(outputs_mask, dim=0), spgen1 + spgen2 + spgen3 + spgen4
