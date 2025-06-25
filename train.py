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


def train_model(model, optimizer, scheduler, num_epochs=5):
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
            running_loss_hint = []
            running_corrects_hint = []

            loaders = dataloaders[phase]
            iters = [iter(loader) for loader in loaders]
            num_batches = sum(len(loader) for loader in loaders)
            finished = [False] * len(loaders)
            # print(num_batches)
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _ in tqdm(range(num_batches)):      
                available = [i for i, f in enumerate(finished) if not f]
                if not available:
                    break
                idx = random.choice(available)
                try:
                    batch = next(iters[idx])
                except StopIteration:
                    finished[idx] = True
                    continue

                for n, value in model.image_encoder.named_parameters():
                    if f"domain{idx}" in n:
                        value.requires_grad = True
                    elif f"domain_all" in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = False

                _, img, labels, img_id = batch

                img = Variable(img.cuda())
                labels = Variable(labels.cuda())

                label_hint = F.interpolate(labels, scale_factor=0.25, mode='nearest')
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                pred_mask, pred_hint = model(x=img, domain_seq=idx)

                pred_mask = torch.sigmoid(pred_mask)
                pred_hint = torch.sigmoid(pred_hint)

                loss1 = mask_loss(pred_mask, labels)
                score_mask1 = iou_metric(pred_mask, labels)

                loss2 = hint_loss(pred_hint, label_hint)

                score_mask2 = acc_metric(pred_hint, label_hint)
                score_mask2 = torch.mean(score_mask2)

                loss = loss1 + 0.5*loss2

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss_mask.append(loss1.item())
                running_corrects_mask.append(score_mask1.item())
                running_loss_hint.append(loss2.item()*0.5)
                running_corrects_hint.append(score_mask2.item())
             

            epoch_mask_loss = np.mean(running_loss_mask)
            epoch_mask_iou = np.mean(running_corrects_mask)
            epoch_hint_loss = np.mean(running_loss_hint)
            epoch_hint_acc = np.mean(running_corrects_hint)
            
            print('{} Mask Loss: {:.4f} Mask IoU: {:.4f} hint Loss: {:.4f} hint Acc: {:.4f} Total Loss: {:.4f}'.format(
                phase, np.mean(epoch_mask_loss), np.mean(epoch_mask_iou),
                np.mean(epoch_hint_loss), np.mean(epoch_hint_acc),
                np.mean(epoch_mask_loss)+np.mean(epoch_hint_loss)))
            
            Loss_list[phase].append(epoch_mask_loss)
            Accuracy_list[phase].append(epoch_mask_iou)

            # save parameters
            if phase == 'valid' and epoch_mask_loss <= best_loss:
                best_loss = epoch_mask_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/UNSAM_{args.domain_num}_B_{epoch}.pth')
                counter = 0
            elif phase == 'valid' and epoch_mask_loss > best_loss:
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
    parser.add_argument('--dataset', type=str,default='ALL', help='MoNuSeg-2018, DSB-2018, SegPC, CryoNuSeg, TNBC')
    parser.add_argument('--sam_pretrain', type=str,default='pretrain/sam_vit_b_01ec64.pth')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--size', type=str,default='B', help='')
    parser.add_argument('--domain_num', type=int,default=3, help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'outputs/',exist_ok=True) 

    jsonfile1 = f'data/DSB-2018/data_split.json'
    
    with open(jsonfile1, 'r') as f:
        df1 = json.load(f)
        train_size = int(round(len(df1['train']) * args.size, 0))
        print(f'DSB-2018: {train_size}')
        train_set1 = np.random.choice(df1['train'],train_size,replace=False)

    jsonfile2 = f'data/MoNuSeg-2018/data_split.json'
    
    with open(jsonfile2, 'r') as f:
        df2 = json.load(f)
        train_size = int(round(len(df2['train']) * args.size, 0))
        print(f'MoNuSeg: {train_size}')
        train_set2 = np.random.choice(df2['train'],train_size,replace=False)

    jsonfile3 = f'data/TNBC/data_split.json'
    
    with open(jsonfile3, 'r') as f:
        df3 = json.load(f)
        train_size = int(round(len(df3['train']) * args.size, 0))
        print(f'TNBC: {train_size}')
        train_set3 = np.random.choice(df3['train'],train_size,replace=False)

    val_files1 = df1['valid'] 
    train_files1 = list(train_set1) 

    val_files2 = df2['valid']
    train_files2 = list(train_set2)

    val_files3 = df3['valid']
    train_files3 = list(train_set3)

    
    train_dataset1 = BinaryLoader("mask_1024", train_files1, A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    
    val_dataset1 = BinaryLoader("mask_1024", val_files1, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_dataset2 = BinaryLoader("mask_1024", train_files2, A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    
    val_dataset2 = BinaryLoader("mask_1024", val_files2, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_dataset3 = BinaryLoader("mask_1024", train_files3, A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    
    val_dataset3 = BinaryLoader("mask_1024", val_files3, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader1 = torch.utils.data.DataLoader(dataset=val_dataset1, batch_size=1)

    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader2 = torch.utils.data.DataLoader(dataset=val_dataset2, batch_size=1)

    train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader3 = torch.utils.data.DataLoader(dataset=val_dataset3, batch_size=1)
    
    dataloaders = {'train':[train_loader1, train_loader2, train_loader3],'valid':[val_loader1, val_loader2, val_loader3]}

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

    pretrain_dict = torch.load(args.sam_pretrain)
    selected_dict = {k: v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.load_state_dict(selected_dict, strict=False)

    selected_dict = {k: v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    model.load_state_dict(selected_dict, strict=False)

    selected_dict = {k: v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    model.load_state_dict(selected_dict, strict=False)


    model = model.cuda()

    for n, value in model.image_encoder.named_parameters():
        if f"domain" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False

    trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
    )

    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')

    total_params = sum(
    param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    print('Ratio = ' + str(trainable_params/total_params) + '%')
        
    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss(0.8) # nn.CrossEntropyLoss()
    hint_loss = nn.BCELoss() # nn.CrossEntropyLoss()
    accuracy_metric = BinaryIoU()

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
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
