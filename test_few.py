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
from PIL import Image,ImageDraw,ImageFilter
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure,color
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from skimage import filters, morphology
from torchvision import transforms
from MAN import MAB
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# import cv2
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap.squeeze(0) * 255).astype(np.uint8)
    # 获取颜色映射并应用
    cmap = plt.get_cmap('jet')
    scoremap = cmap(scoremap.astype(float) / 255.0)[:, :, :3]  # 取RGB通道
    # 将彩色scoremap转换为uint8
    scoremap = (scoremap * 255).astype(np.uint8)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def main():
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
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()


    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))



    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

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


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):

    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    anomaly_seg_map = []
    seg_score_map_zero = []
    seg_score_map_few= []
    i=0
    for (image, y, mask) in tqdm(test_loader):

        # <class 'torch.Tensor'>
        # print(type(image))
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
        # print(mask.shape)
        #torch.Size([1, 1, 240, 240]
        #torch.Size([1, 3, 14, 14])
        # print(encoded_image.shape)
        # print(image.shape)[1, 3, 240, 240]
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)

                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                # print(score_map_few.shape)#(1, 240, 240)

                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
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
                #需要可视化时放出
                # all_seg_map = score_map_few + score_map_zero
                # anomaly_seg_map.append(all_seg_map)
                #可视化放出
                """
                if y == 1:
                    i += 1

                    # 读取和调整图像大小
                    image = (image.squeeze(0).detach().cpu()).numpy()
                    image = (np.transpose(image, (1, 2, 0))* 255).astype(np.uint8)
                    这里为止"""
                """
                    vis = cv2.cvtColor(cv2.resize(image, (args.img_size, args.img_size)), cv2.COLOR_BGR2RGB)  # RGB
                    # # 归一化异常检测图
                    # anomaly_map_norm = normalize(anomaly_seg_map[i-1])
                    # _, binary_map = cv2.threshold(anomaly_map_norm, 0.7, 1, cv2.THRESH_BINARY)
                    # # 将二值化图像转换为白色（255）和黑色（0）
                    # binary_map = (binary_map.squeeze(0)  * 255).astype(np.uint8)
                    # # 使用 apply_ad_scoremap 函数将归一化后的异常检测图 mask 应用到原始图像 vis 上，生成带有异常区域高亮的可视化图像。
                    # vis1 = apply_ad_scoremap(vis, anomaly_map_norm)
                    # # 将图像从 RGB 颜色空间转换回 BGR 颜色空间，因为 OpenCV 和许多图像处理库默认使用 BGR 颜色空间。
                    # vis2 = cv2.cvtColor(vis1, cv2.COLOR_RGB2BGR)  # BGR
                """
                # 归一化异常检测图
                """可视化放出
                    anomaly_map_norm = normalize(anomaly_seg_map[i - 1])
                    # 使用numpy进行二值化处理
                    binary_map = (anomaly_map_norm > 0.7).astype(np.uint8) * 255
                    # 将二值化图像转换为RGB格式
                    binary_map = np.stack((binary_map,) * 3, axis=-1)
                    # 使用 apply_ad_scoremap 函数将归一化后的异常检测图 mask 应用到原始图像 vis 上，生成带有异常区域高亮的可视化图像。
                    vis1 = apply_ad_scoremap(image, anomaly_map_norm)"""

                    # 构建保存可视化图像的路径 save_vis，路径格式为 save_path/imgs/类别名/子类别名
                    # save_vis = os.path.join('./results', 'onlyRAB', args.obj)

                    #可视化放出
                    # save_vis = os.path.join('./resultsmvfa', 'xr', args.obj)
                    #
                    # filename = args.obj+str(i)
                    # if not os.path.exists(save_vis):  # 检查保存路径是否存在，如果不存在，则创建该路径。
                    #     os.makedirs(save_vis)
                    # # 使用 cv2.imwrite 将处理后的可视化图像保存到指定路径。
                    # save_vis_path = os.path.join(save_vis, filename+'.png')
                    # save_image_path = os.path.join(save_vis, args.obj + '_image' + str(i) + '.png')
                    # save_mask_path = os.path.join(save_vis, args.obj + '_mask' + str(i) + '.png')
                    # save_anomalymask_path = os.path.join(save_vis, args.obj + '_truemask' + str(i) + '.png')
                    # cv2_mask = (mask.squeeze(0).squeeze(0) * 255).byte().cpu().numpy()
                    ##########以下不是
                    # cv2.imwrite(save_vis_path, vis2,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                    # cv2.imwrite(save_image_path, vis,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                    # cv2.imwrite(save_mask_path, cv2_mask)
                    # cv2.imwrite(save_anomalymask_path, binary_map)
                    #可视化放出

                    # plt.imsave(save_vis_path, vis1, format='png')
                    # plt.imsave(save_image_path, image, format='png')
                    # plt.imsave(save_mask_path, cv2_mask, format='png')
                    # plt.imsave(save_anomalymask_path, binary_map, format='png')
                    # print("保存成功")

            else:
                # few-shot, det head
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

                # zero-shot, det head
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
        # print(seg_score_map_few.shape)#(1805, 1, 240, 240)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        # print(segment_scores.shape)

        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:

        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det





if __name__ == '__main__':
    main()


