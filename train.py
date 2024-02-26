import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import BinaryLoader, OneHotLoader
from loss import *
from tqdm import tqdm
import json
from model import UNSAM
from SAM.modeling.image_encoder import DTEncoder
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# torch.set_num_threads(4)


def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_mask = []
            running_corrects_mask = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _, img, labels, _ in tqdm(dataloaders[phase]):      
                # wrap them in Variable
                if torch.cuda.is_available():
    
                    img = Variable(img.cuda())
                    labels = Variable(labels.cuda())
                 
                else:
                    img, labels = Variable(img), Variable(labels)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # if phase == 'train':
                for n, value in model.image_encoder.named_parameters():
                    if f"domain{args.domain_num}" in n:
                        value.requires_grad = True
                    elif f"domain_all" in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = False
                
                pred_mask, pred_hint = model(x=img, domain_seq=args.domain_num)

                pred_mask = torch.sigmoid(pred_mask)
                pred_hint = torch.sigmoid(pred_hint)

                loss1 = criterion_mask(pred_mask, labels)
                score_mask1 = accuracy_metric(pred_mask, labels)
                
                loss2 = hint_loss(pred_hint, labels)
                score_mask2 = accuracy_metric(pred_hint, labels)

                loss = loss1 + loss2

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss_mask.append(loss.item())
                running_corrects_mask.append(score_mask1.item())
             

            epoch_loss = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            
            print('{} Loss 1: {:.4f} IoU 1: {:.4f}'.format(
                phase, np.mean(running_loss_mask), np.mean(running_corrects_mask)))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # if epoch > 10:
            #     torch.save(best_model_wts, f'/raid/newuser/xq/sam_med/outputs/ViT_H/model_decouple_domain_{args.domain_num}_epoch_{epoch}.pth')
            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/ViT_{args.size}/model_decouple_domain_{args.domain_num}_epoch_{epoch}.pth')
                counter = 0
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'valid':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    
    

    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='TNBC', help='MoNuSeg-2018, DSB-2018, SegPC, CryoNuSeg, TNBC')
    parser.add_argument('--sam_pretrain', type=str,default='pretrain/sam_vit_h_4b8939.pth')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--size', type=str,default='H', help='')
    parser.add_argument('--domain_num', type=int,default=1, help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoches')
    args = parser.parse_args()

    # os.makedirs(f'outputs/',exist_ok=True)

    args.jsonfile = f'/data/xq/sam_med/datasets/{args.dataset}/data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    val_files = df['valid']
    train_files = df['train']
    
    train_dataset = BinaryLoader(args.dataset, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    val_dataset = BinaryLoader(args.dataset, val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}

    if args.size == 'H':
        vit = DTEncoder(
                depth=32,
                embed_dim=1280,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[7, 15, 23, 31],
                window_size=14,
                out_chans=256,
                domain_num=args.domain_num
            )
    elif args.size == 'L':
        vit = DTEncoder(
                depth=24,
                embed_dim=1024,
                img_size=1024,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=16,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[5, 11, 17, 23],
                window_size=14,
                out_chans=256,
                domain_num=args.domain_num
            )
    else:
        vit = DTEncoder(
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
                domain_num=args.domain_num
            )
    

    model = UNSAM(image_encoder=vit)

    encoder_dict = torch.load(args.sam_pretrain)
    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    model.load_state_dict(pre_dict, strict=False)

    # model.mask_decoder.load_state_dict(model.mask_decoder.state_dict(), strict=False)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.sam_pretrain), strict=True)
    # model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.sam_pretrain).items()}, strict=True)

    # for i in range(12):
            
    #     model.image_encoder.blocks[i].p_domain3.load_state_dict(model.image_encoder.blocks[i].p_domain1.state_dict())
    #     model.image_encoder.blocks[i].attn.deep_QKV_embeddings_domain3.data = model.image_encoder.blocks[i].attn.deep_QKV_embeddings_domain1.data

    model = model.cuda()
        
    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss(0.8) # nn.CrossEntropyLoss()
    hint_loss = nn.BCELoss() # nn.CrossEntropyLoss()
    accuracy_metric = BinaryIoU()

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    Loss_list, Accuracy_list = train_model(model, mask_loss, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')