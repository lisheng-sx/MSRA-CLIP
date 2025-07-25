import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from MAN import MAB
from RAB import Residual_Attention_Block
#计算FLOPS
from thop import profile

import time
from concurrent.futures import ThreadPoolExecutor

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Histopathology':-3}

#设置随机种子：确保实验的可重复性，通过设置随机数生成器的种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 # 用来保存训练以及验证过程中信息
results_file = "resultsfew{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
def main():
    #定义主函数：使用argparse模块解析命令行参数，允许用户在运行时指定模型名称、数据路径、批量大小等。
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()
    setup_seed(args.seed)
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()


    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = True


    # seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    # det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999),weight_decay=args.weight_decay)
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999),weight_decay=args.weight_decay)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)
    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

    for epoch in range(args.epoch):
            print('epoch ', epoch, ':')
            with open(results_file, 'a') as f:
                f.write(f"Epoch: {epoch}]\n")
            loss_list = []
            for (image, gt, label) in train_loader:
                image = image.to(device)
                with torch.cuda.amp.autocast():
                    image1 = mab(image)
                    image2 = rab(image)
                    concat = torch.cat((image1, image2), dim=1)
                    channel_reducer = nn.Conv2d(6, 3, kernel_size=1).half().to(device)

                    image3 = channel_reducer(concat)
                    image = image3 + image
                    _, seg_patch_tokens, det_patch_tokens = model(image)
                    seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                    det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                    # det loss
                    det_loss = 0
                    image_label = label.to(device)
                    for layer in range(len(det_patch_tokens)):
                        det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score = torch.mean(anomaly_map, dim=-1)
                        det_loss += loss_bce(anomaly_score, image_label)
                    if CLASS_INDEX[args.obj] > 0:
                        seg_loss = 0
                        mask = gt.squeeze(0).to(device)
                        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                        for layer in range(len(seg_patch_tokens)):
                            seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                            anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                            B, L, C = anomaly_map.shape
                            H = int(np.sqrt(L))
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            seg_loss += loss_focal(anomaly_map, mask)
                            seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                        loss = seg_loss + det_loss
                        loss.requires_grad_(True)
                        seg_optimizer.zero_grad()
                        det_optimizer.zero_grad()
                        loss.backward()
                        seg_optimizer.step()
                        det_optimizer.step()

                    else:
                        loss = det_loss
                        loss.requires_grad_(True)
                        det_optimizer.zero_grad()
                        loss.backward()
                        det_optimizer.step()

                    loss_list.append(loss.item())


            print("Loss: ", np.mean(loss_list))

            seg_features = []
            det_features = []
            for image in support_loader:
                image = image[0].to(device)
                with torch.no_grad():

                    _, seg_patch_tokens, det_patch_tokens = model(image)
                    seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                    det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                    seg_features.append(seg_patch_tokens)
                    det_features.append(det_patch_tokens)
            seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
            det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]

            result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
            if result > best_result:
                best_result = result
                with open(results_file, 'a') as f:
                    f.write(f"Best result\n")
                print("Best result\n")
                if args.save_model == 1:
                    ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapters': model.det_adapters.state_dict()},
                                ckp_path)



#定义测试函数：用于评估模型在测试集上的表现，包括计算分割和检测的得分。
def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []

    seg_score_map_zero = []
    seg_score_map_few= []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]
            if CLASS_INDEX[args.obj] > 0:

                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)


            else:
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        with open(results_file, 'a') as f:
            f.write(f"Epoch pAUC: {round(seg_roc_auc, 4)}\n")
            f.write(f"Epoch AUC: {round(roc_auc_im, 4)}\n")
        return seg_roc_auc + roc_auc_im

    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        with open(results_file, 'a') as f:
            f.write(f"Epoch AUC: {round(img_roc_auc_det, 4)}\n")
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det



if __name__ == '__main__':
    mab = MAB(n_feats=3)
    mab.to(device)
    rab = Residual_Attention_Block(in_channels=3, out_channels=3, bias=True)
    rab.to(device)
    main()


