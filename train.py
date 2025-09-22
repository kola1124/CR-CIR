import os 
import argparse
import logging
import warnings 
import random


import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
from tqdm import tqdm
from utils.meter import AverageMeter
import time
import open_clip 
import utilss
import datasets
import model
#import models
import test
import torch.nn as nn
from torch.cuda.amp import autocast as autocast, GradScaler
from torchinfo import summary

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from sklearn.mixture import GaussianMixture


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

torch.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_split', default = 'val-split')
parser.add_argument('--fashioniq_path', default = '../data_dqu/FashionIQ')
parser.add_argument('--shoes_path', default = '../data_dqu/Shoes')
parser.add_argument('--fashion200k_path', default = '../data_dqu/Fashion200K')
parser.add_argument('--cirr_path', default = '../data_dqu/CIRR')
#'val-split'
parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=512)

parser.add_argument('--seed', type=int, default=42)   
parser.add_argument('--lr', type=float, default=1e-4) 
parser.add_argument('--clip_lr', type=float, default=1e-6) 

parser.add_argument('--backbone', type=str, default='ViT-B-16')

parser.add_argument('--lr_decay', type=int, default=10)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=10)  #学习率调整过程
parser.add_argument('--tolerance_epoch', type=int, default=100)

 
parser.add_argument('--model_dir', default='./checkpoints',
                    help="Directory containing params.json")

parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--i', type=str, default='0')

parser.add_argument("--gpu", type=int, default=0, help="id")
parser.add_argument("--use_hog", type=bool, default=False, help='W')
parser.add_argument("--mask_ratio", type=float, default=0.3)
parser.add_argument("--alpha", type=float, default=0.1)

args = parser.parse_args()

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
        
def load_dataset():
    #clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='/home/mail/2022s2/s230201705/conference_new/DQU-CIR-main/src/laion2B-s32B-b79K/open_clip_pytorch_model.bin')
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='/src/CLIP-ViT-B-16-laion2B-s34B-b88K /open_clip_pytorch_model.bin')
    
    print('preprocess_train', preprocess_train)
    
    print('preprocess_val', preprocess_val)
    if args.dataset in ['dress', 'shirt', 'toptee']:
        print('Loading FashionIQ-{} dataset'.format(args.dataset))
        print('fashioniq_spilt', args.fashioniq_split)
        #'../data/FashionIQ'    #'dress'    
        fashioniq_dataset = datasets.fashioniqall(path = args.fashioniq_path, category = args.dataset, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        print('length', len(fashioniq_dataset))
        #fashioniq_dataset_shirt = datasets.fashioniqall(path = args.fashioniq_path, category = 'shirt', transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        #fashioniq_dataset_ = datasets.fashioniqall(path = args.fashioniq_path, category = 'toptee', transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        #print('Readingingi',type(fashioniq_dataset))
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.shoesshoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'cirr':
        print('Reading cirr')
        cirr_dataset = datasets.cirrcirr(path = args.cirr_path, transform = [preprocess_train, preprocess_val])
        return [cirr_dataset]
    elif args.dataset == 'fashion200k':
        print('Reading fashion200k')
        fashion200k_dataset = datasets.Fashion200k(path = args.fashion200k_path, split = 'train', transform = [preprocess_train, preprocess_val])
        fashion200k_testset = datasets.Fashion200k(path = args.fashion200k_path, split = 'test', transform = [preprocess_train, preprocess_val])
        return [fashion200k_dataset, fashion200k_testset]


def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    DQU_CIR_model = model.DQU_CIR(args.hidden_dim, args.dropout_rate, alpha=args.alpha)
    #DQU_CIR_model = models.DQU_CIR(args.hidden_dim, args.dropout_rate)
    DQU_CIR_model.cuda()

    params = list(DQU_CIR_model.named_parameters())
    param_group = [
    #     {
    #     'params': [p for n, p in params if 'clip' in n or 'caption' in n],
    #     'lr': args.clip_lr
    # },
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr},
    #     {
    #     'params': [p for n, p in params if 'clip' not in n and 'caption' not in n],
    #     'lr': args.lr
    # },
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)
    
    return DQU_CIR_model, optimizer

def train(model, optimizer, dataloader, scaler, meters, epoch):
    model.train()
    model.apply(set_bn_eval)
    
    #summ = []
    #loss_avg = utils.RunningAverage()
    # with tqdm(total=len(dataloader)) as t:
    for i, data in enumerate(dataloader):
            #print('data.keys', data.keys()) #dict_keys(['source_img_data', 'target_img_data', 'mod', 'textual_query', 'visual_query'])
        target_img = data['target_img_data'].cuda()
        textual_query = data['textual_query']
        visual_query = data['visual_query'].cuda()
    
        optimizer.zero_grad()
        with autocast():
            
            ret = model.compute_loss(textual_query, visual_query, target_img, data)
            
        
        total_loss = sum([v for k, v in ret.items() if "loss" in k])
        batch_size = data['target_img_data'].shape[0]
        meters['loss'].update(total_loss.item(), batch_size)
        meters['hog_loss'].update(ret.get('hog_loss', 0), batch_size)
        meters['nce_loss'].update(ret.get('nce_loss', 0), batch_size)
        meters['kl_loss'].update(ret.get('kl_loss', 0), batch_size)
        

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        
        if i % 100 == 0:
            #print('loss={:05.3f}'.format(meters['nce_loss'].avg))
            print("Epoch: {}/{} Iteration:{}/{} total_loss: {:.3f} nce_loss: {:.3f} hog_loss: {:.3f} kl_loss: {:.3f}".format(epoch+1, args.num_epochs, i, len(dataloader), meters['loss'].avg, meters['nce_loss'].avg, meters['hog_loss'].avg, meters['kl_loss'].avg))
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        # t.update()

def train_and_evaluate(model, optimizer, dataset_list):
    torch.set_printoptions(threshold=np.inf)
    
    np.set_printoptions(threshold=np.inf)
    if args.dataset == 'fashion200k':
        fashion200k_testset = dataset_list.pop(-1)
    trainloader = dataloader.DataLoader(dataset_list[0],
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers=args.num_workers)

    
    meters ={'loss': AverageMeter(),
              'nce_loss': AverageMeter(),
              'hog_loss': AverageMeter(),
              'kl_loss':AverageMeter()
            #  'nce_loss_tse': AverageMeter(),
             
            #  'sdm_loss': AverageMeter(),
            #  'sdm_loss_tse': AverageMeter()
             }
    
    best_score = float('-inf')
    tolerance = 0
    scaler = GradScaler()
    epoches = args.num_epochs
    # print('Test first')
    # current_score = 0
    # # if tolerance < args.tolerance_epoch:
    
    if args.dataset in ['shoes']:
        with torch.no_grad():
            t_shoes = test.test_shoes(args, model, dataset_list[0], args.dataset)
            logging.info(t_shoes)
            
    if args.dataset in ['dress', 'shirt', 'toptee']:
        with torch.no_grad():
                    #这个dataset_list就是dataset_list
            t_dress, t_dresstxt, t_all = test.test_dress(args, model, dataset_list[0], args.dataset)
            t_toptee, t_topteetxt, _ = test.test_toptee(args, model, dataset_list[0], args.dataset)
            t_shirt, t_shirttxt, _ = test.test_shirt(args, model, dataset_list[0], args.dataset)
        logging.info(t_dress)
        #logging.info(t_dresstxt)
        
        logging.info(t_toptee)
        #logging.info(t_topteetxt)
        
        logging.info(t_shirt)
        #logging.info(t_shirttxt)
        # logging.info(t_tse)
        # logging.info(t_all)
        #current_score = current_score

            

    for epoch in range(epoches):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
            
            
            
        tolerance = tolerance + 1
        if epoch != 0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div
        
        
        np.set_printoptions(threshold=np.inf)   
        torch.set_printoptions(threshold=np.inf)
     
        model.train()
     
        
        #print('label_hat', label_hat)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'weights/model_weights_shoes.pth')
        train(model, optimizer, trainloader, scaler, meters, epoch)
        
        #print("Epoch: {}/{} loss: {}".format(epoch+1, epoches, ))
        print('Test Begin.........')
        current_score = 0
        if tolerance < args.tolerance_epoch:
            if args.dataset in ['dress', 'shirt', 'toptee']:
                with torch.no_grad():
                    #这个dataset_list就是dataset_list
                    t_dress, t_dresstxt, t_dressall = test.test_dress(args, model, dataset_list[0], args.dataset)
                    t_toptee, t_topteetxt, t_topteeall = test.test_toptee(args, model, dataset_list[0], args.dataset)
                    t_shirt, t_shirttxt, t_shirtall = test.test_shirt(args, model, dataset_list[0], args.dataset)
                    
                logging.info(t_dress)
               
        
                logging.info(t_toptee)
               
        
                logging.info(t_shirt)
          
                current_score = current_score

            elif args.dataset in ['shoes']:
                with torch.no_grad():
                    #这个dataset_list就是dataset_list
                    t_shoes = test.test_shoes(args, model, dataset_list[0], args.dataset)
                    
                logging.info(t_shoes)
             
                current_score = current_score
            elif args.dataset in ['fashion200k']:
                t = test.test_fashion200k_dataset(args, model, fashion200k_testset)
                logging.info(t)
                current_score = current_score + t[0][1] + t[1][1] + t[2][1]
            
            elif args.dataset in ['cirr']:
                t, t_tse, t_all = test.test_cirr_valset(args, model, dataset_list[0])
                logging.info(t)
                # logging.info(t_tse)
         
        else:
            break

def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


if __name__ == '__main__':
    
    # Load the parameters from json file
    print(open_clip.__file__)
    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    utilss.set_logger(os.path.join(args.model_dir, '{}_{}_train.log'.format(args.dataset, args.i)))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    dataset_list = load_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))

    train_and_evaluate(model, optimizer, dataset_list)
