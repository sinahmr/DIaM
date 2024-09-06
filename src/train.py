import argparse
import yaml
import os
import time
import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
import torch.distributed as dist
from .model.pspnet import get_model
from torch.nn.parallel import DistributedDataParallel as DDP
from .dataset.datasetV1 import get_val_loader
from .util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg
from .model.pspnet import get_model
from .classifierV1 import Classifier
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import sys



def main(rank: int, world_size: int, args: argparse.Namespace) -> None:
    # 1. Load Configuration
    torch.cuda.empty_cache()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    setup(args, rank, world_size)
    # 2. Setup Device (GPU or CPU)
    device = torch.device('cuda:{}'.format(args.gpus) if torch.cuda.is_available() and args.gpus != -1 else 'cpu')
    #setup the args
    print(f"==> Running setup script")
    # setup(args, rank, world_size)
    setup_seed(cfg['EVALUATION']['manual_seed'])
    # 3. Datasets and DataLoaders
    # ========== Data  ==========
    val_loader = get_val_loader(cfg, args)
    print(f'rank in raw....',{rank})
    
    # ========== Model  ==========
    model = get_model(cfg, args).to(0)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[0])
    
    # cfg: dict, args: argparse.Namespace, run_id=None
    root = get_model_dir(cfg, args)

    print("=> Creating the model")
    if cfg['EVALUATION']['ckpt_used'] is not None:
        filepath = os.path.join(root, f"{cfg['EVALUATION']['ckpt_used']}.pth")
        assert os.path.isfile(filepath), filepath
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Test  ==========
    print('starting validation in ptrain file....')
   
    validateNow(args=args, val_loader=val_loader, model=model, cfg=cfg)
    # cleanup()

def validateNow(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP,cfg: dict) -> Tuple[torch.tensor, torch.tensor]:
    print('\n==> Start testing ({} runs)'.format(cfg['EVALUATION']['n_runs']), flush=True)
    random_state = setup_seed(cfg['EVALUATION']['manual_seed'], return_old_state=True)
    device = torch.device('cuda:{}'.format(dist.get_rank()))
    model.eval()

    c = model.module.bottleneck_dim
    h = model.module.feature_res[0]
    w = model.module.feature_res[1]
    print(f'channel .....{c}')
    nb_episodes = len(val_loader) if cfg['EVALUATION']['test_num'] == -1 else int(cfg['EVALUATION']['test_num'] / cfg['EVALUATION']['batch_size_val'])
    runtimes = torch.zeros(cfg['EVALUATION']['n_runs'])
    base_mIoU, novel_mIoU = [torch.zeros(cfg['EVALUATION']['n_runs'], device=device) for _ in range(2)]

    # ========== Perform the runs  ==========
    for run in range(cfg['EVALUATION']['n_runs']):
        print('Run', run + 1, 'of', cfg['EVALUATION']['n_runs'])
        # The order of classes in the following tensors is the same as the order of classifier (novels at last)
        cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
        # print(f'CLS_INTERSECTION.....{cls_intersection.shape}')
        print(f'CLS_INTERSECTION.....{args.num_classes_tr}')
        # print(f'CLS_INTERSECTION.....{args.num_classes_val}')
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)
        # print(f'CLS_INTERSECTION.....{args.num_classes_tr} + {args.num_classes_val}')
        # print(f'CLS_INTERSECTION.....{args.num_classes_tr}')
        runtime = 0
        features_s, gt_s = None, None
        if not cfg['EVALUATION']['generate_new_support_set_for_each_task']:
            with torch.no_grad():
                spprt_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if cfg['EVALUATION']['test_num'] == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                # print(f'support shape model extractor before........{spprt_imgs.shape}')
                features_s = model.module.extract_features(spprt_imgs).detach().view((args.num_classes_val, cfg['EVALUATION']['shot'], c, h, w))
                # print(f'features extracted here is...{features_s.shape}')
                # print(f'NUM_CLASSES_IN_VAL....{args.num_classes_val}')
                gt_s = s_label.view((args.num_classes_val, cfg['EVALUATION']['shot'], cfg['DATA']['image_size'], cfg['DATA']['image_size']))
                print(f'features extracted here is...{gt_s.shape}')
        # sys.exit(1)
        for _ in tqdm(range(nb_episodes), leave=True):
            t0 = time.time()
            with torch.no_grad():
                try:
                    loader_output = next(iter_loader)
                except (UnboundLocalError, StopIteration):
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                qry_img, q_label, q_valid_pix, img_path = loader_output
                # hape of query image in train... torch.Size([20, 3, 417, 417])
                # print(f'shape of query image in train...', qry_img.shape)
                # # shape of label image in train... torch.Size([20, 3, 417, 417])
                # print(f'shape of label image in train...', qry_img.shape)
                # # shape of VALID_PIXEL in train... torch.Size([20, 417, 417])
                # print(f'shape of VALID_PIXEL in train...', q_valid_pix.shape)
                image_height = qry_img.shape[1] # Height
                image_width = qry_img.shape[2]  # Width
                # print(f"Image Height: {image_height}, Width: {image_width}")
                qry_img = qry_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)
                features_q = model.module.extract_features(qry_img).detach().unsqueeze(1)
                # FEATURE AFTER EXTRACTION IS.....torch.Size([20, 1, 512, 53, 53])
                # print(f'FEATURE AFTER EXTRACTION IS.....{features_q.shape}')
                valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
                # VALID PIXEL SHAPE HERE.....torch.Size([20, 1, 417, 417])
                # print(f'VALID PIXEL SHAPE HERE.....{valid_pixels_q.shape}')
                gt_q = q_label.unsqueeze(1)
                # GROUND LABEL SHAPE.....torch.Size([20, 1, 417, 417])
                print(f'GROUND LABEL SHAPE.....{gt_q.shape}')

                query_image_path_list = list(img_path)
                if cfg['EVALUATION']['generate_new_support_set_for_each_task']:
                    spprt_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)
                    features_s = model.module.extract_features(spprt_imgs).detach().view((args.num_classes_val, args.shot, c, h, w))
                    gt_s = s_label.view((args.num_classes_val, cfg['EVALUATION']['shot'], cfg['DATA']['image_size'], cfg['DATA']['image_size']))
                    print(f'GROUND TRUTH SHAPE INITIAlIZED.....{gt_s.shape}')
                    # sys.exit(1)
            # =========== Initialize the classifier and run the method ===============
            # print("SHAPE OF BASE WEIGHT BEFORE TRANSPOSE...",model.module.classifier.weight.shape )
            if len(model.module.classifier.weight.shape) == 2:
                base_weight = model.module.classifier.weight.detach().clone().mT  # or .T
            else:
                num_dims = model.module.classifier.weight.ndim  # Get the number of dimensions
                base_weight = model.module.classifier.weight.detach().clone().permute(*torch.arange(num_dims - 1, -1, -1))
                feature_dim = model.module.get_feature_dim()
                # print(f' THE FEATURE DIMENSION IS ....{feature_dim}')
                # Now you can use base_weight
                # print("NEW SHAPE AFTER TRANSPOSE...", base_weight.shape)
            # print(f'the feature dimension from the model is {num_dims}')
            # base_weight = model.module.classifier.weight.detach().clone().T
            base_bias = model.module.classifier.bias.detach().clone()
            # (self, configer, backbone, feature_dim, num_prototypes, model):
            # print("INITIAL SHAPE features_q...", features_q.shape)
            # print("INITIAL SHAPE gt_s...", gt_s.shape)
            print(f"number of novel classes in training before classifier passed to...{args.num_novel_classes}")
            classifier = DIaMClassifier(args, base_weight, base_bias, n_tasks=features_q.size(0), cfg=cfg, backbone=model,features_s=features_s,gt_s=gt_s,num_novel_classes= args.num_novel_classes,feature_dim=feature_dim)
            # print(f'After the training here..')
            # classifier = Classifier(args, base_weight, base_bias, n_tasks=features_q.size(0))
            print(f'Before the init_prototype section feaures support {features_s.shape} and ground support {gt_s.shape}')
            classifier.init_prototypes(features_s, gt_s)

            # INITIAL SHAPE features_s... torch.Size([20, 1, 512, 53, 53])
            # print("INITIAL SHAPE features_s...", features_s.shape)
            # INITIAL SHAPE features_q... torch.Size([20, 1, 512, 53, 53])
            # print("INITIAL SHAPE features_q...", features_q.shape)
            # INITIAL SHAPE gt_s... torch.Size([20, 1, 417, 417])
            print("INITIAL SHAPE gt_s...", gt_s.shape)
            # INITIAL SHAPE valid_pixels_q... torch.Size([20, 1, 417, 417])
            # print("INITIAL SHAPE valid_pixels_q...", valid_pixels_q.shape)
            # sys.exit(1)
            classifier.optimize(features_s, features_q, gt_s, valid_pixels_q)
            # print(f'After optimization here....')
            runtime += time.time() - t0

            # =========== Perform inference and compute metrics ===============
            logits = classifier.get_logits(features_q).detach()
            probas = classifier.get_probas(logits)
            print(f'Probability ....{probas.shape} and ground truth shape ... {gt_q.shape}')
            # Probability ....torch.Size([20, 1, 20, 81, 53]) and ground truth shape ... torch.Size([20, 1, 417, 417])
            intersection, union, target = fast_intersection_and_union(probas, gt_q)  # [batch_size_val, 1, num_classes]
            print(f"Intersection shape: {intersection.shape}")
            print(f"Union shape: {union.shape}")
            print(f"Target shape: {target.shape}")
            intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
            
            print(f"Squeezed Intersection shape: {intersection.shape}")
            print(f"Squeezed Union shape: {union.shape}")
            print(f"Squeezed Target shape: {target.shape}")

            try:
                # cls_intersection += intersection.sum(0)
                # cls_union += union.sum(0)
                # cls_target += target.sum(0)
                cls_intersection[args.num_classes_tr:] += intersection.sum(0)  # Change 2: Update the relevant part
                cls_union[args.num_classes_tr:] += union.sum(0)                # Change 2: Update the relevant part
                cls_target[args.num_classes_tr:] += target.sum(0)  
            except RuntimeError as e:
                print(f"Error in cls_intersection update: {e}. Skipping this operation.")
                print(f"Error in cls_intersection update: {e}. Skipping this operation.")
                print(f"cls_intersection shape: {cls_intersection.shape}")
                print(f"cls_union shape: {cls_union.shape}")
                print(f"cls_target shape: {cls_target.shape}")
            # cls_intersection += intersection.sum(0)
            # cls_union += union.sum(0)
            # cls_target += target.sum(0)

        base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
        for i, class_ in enumerate(val_loader.dataset.all_classes):
            if cls_union[i] == 0:
                continue
            IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
            print("Class {}: \t{:.4f}".format(class_, IoU))
            if class_ in val_loader.dataset.base_class_list:
                sum_base_IoU += IoU
                base_count += 1
            elif class_ in val_loader.dataset.novel_class_list:
                sum_novel_IoU += IoU
                novel_count += 1

        avg_base_IoU, avg_novel_IoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
        print('Mean base IoU: {:.4f}, Mean novel IoU: {:.4f}'.format(avg_base_IoU, avg_novel_IoU), flush=True)

        base_mIoU[run], novel_mIoU[run] = avg_base_IoU, avg_novel_IoU
        runtimes[run] = runtime

    agg_mIoU = (base_mIoU.mean() + novel_mIoU.mean()) / 2
    print('==>')
    print('Average of base mIoU: {:.4f}\tAverage of novel mIoU: {:.4f} \t(over {} runs)'.format(
        base_mIoU.mean(), novel_mIoU.mean(), args.n_runs))
    print('Mean --- {:.4f}'.format(agg_mIoU), flush=True)
    print('Average runtime / run --- {:.1f}\n'.format(runtimes.mean()))

    resume_random_state(random_state)
    return agg_mIoU

def evaluate(model, data_loader, device, cfg):
    return 0


if __name__ == "__main__":
    # Original tensor

    parser = argparse.ArgumentParser(description='DIaM Training and Testing Script')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--split', type=int, help='Data split to use')
    parser.add_argument('--shot', type=int, help='Number of shots')
    parser.add_argument('--gpus', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    args = parser.parse_args()
    # if args.debug:
    #     args.test_num = 64
    #     args.n_runs = 2

    world_size = len(str(args.gpus))
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    main(0, world_size, args)

