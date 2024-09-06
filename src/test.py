import argparse
import os
import yaml
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .classifierV1 import Classifier
from .dataset.datasetV1 import get_val_loader
from .model.pspnet import get_model
from .util import get_model_dir, fast_intersection_and_union, setup_seed, resume_random_state, find_free_port, setup, \
    cleanup, get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)
# data/coco/val2014/COCO_val2014_000000054091.jpg
# data/coco/val2014/COCO_val2014_000000039115.jpg
def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")
    # Access the 'config' argument directly
    print(f'NAME SPACE....{args}') 
    cfg = args
    setup(args, rank, world_size)
    
    setup_seed(cfg['EVALUATION']['manual_seed'])
    # setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(cfg, args)
    # val_loader = get_val_loader(args)

    # ========== Model  ==========
    model = get_model(cfg,args).to(0)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[0])

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
    validate(args=args, val_loader=val_loader, model=model, cfg=cfg)
    cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP,cfg: dict) -> Tuple[torch.tensor, torch.tensor]:
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
        cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
        cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

        runtime = 0
        features_s, gt_s = None, None
        if not cfg['EVALUATION']['generate_new_support_set_for_each_task']:
            with torch.no_grad():
                spprt_imgs, s_label = val_loader.dataset.generate_support([], remove_them_from_query_data_list=True)
                nb_episodes = len(val_loader) if cfg['EVALUATION']['test_num'] == -1 else nb_episodes  # Updates nb_episodes since some images were removed by generate_support
                spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                s_label = s_label.to(device, non_blocking=True)
                features_s = model.module.extract_features(spprt_imgs).detach().view((args.num_classes_val, args.shot, c, h, w))
                gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))
                print(f'running here...')

        for _ in tqdm(range(nb_episodes), leave=True):
            t0 = time.time()
            with torch.no_grad():
                try:
                    loader_output = next(iter_loader)
                except (UnboundLocalError, StopIteration):
                    iter_loader = iter(val_loader)
                    loader_output = next(iter_loader)
                qry_img, q_label, q_valid_pix, img_path = loader_output

                qry_img = qry_img.to(device, non_blocking=True)
                q_label = q_label.to(device, non_blocking=True)
                features_q = model.module.extract_features(qry_img).detach().unsqueeze(1)
                valid_pixels_q = q_valid_pix.unsqueeze(1).to(device)
                gt_q = q_label.unsqueeze(1)
                print(f'running here. images..')
                query_image_path_list = list(img_path)
                if cfg['EVALUATION']['generate_new_support_set_for_each_task']:
                    spprt_imgs, s_label = val_loader.dataset.generate_support(query_image_path_list)
                    spprt_imgs = spprt_imgs.to(device, non_blocking=True)
                    s_label = s_label.to(device, non_blocking=True)
                    features_s = model.module.extract_features(spprt_imgs).detach().view((args.num_classes_val, acfg['EVALUATION']['shot'], c, h, w))
                    gt_s = s_label.view((args.num_classes_val, cfg['EVALUATION']['shot'], cfg['DATA']['image_size'], cfg['DATA']['image_size']))

            # =========== Initialize the classifier and run the method ===============
            base_weight = model.module.classifier.weight.detach().clone().T
            base_bias = model.module.classifier.bias.detach().clone()
            print(f'running classifier...')
            # classifier = DIaMClassifier(args, base_weight, base_bias, n_tasks=features_q.size(0), cfg=cfg, backbone=model,features_s=features_s,gt_s=gt_s,num_novel_classes= args.num_novel_classes,feature_dim=feature_dim)
            classifier = Classifier(args, base_weight, base_bias, n_tasks=features_q.size(0), cfg=cfg, backbone=model)
            classifier.init_prototypes(features_s, gt_s)
            classifier.compute_pi(features_q, valid_pixels_q, gt_q)  # gt_q won't be used in optimization if pi estimation strategy is self or uniform
            classifier.optimize(features_s, features_q, gt_s, valid_pixels_q)

            runtime += time.time() - t0

            # =========== Perform inference and compute metrics ===============
            logits = classifier.get_logits(features_q).detach()
            probas = classifier.get_probas(logits)

            intersection, union, target = fast_intersection_and_union(probas, gt_q)  # [batch_size_val, 1, num_classes]
            intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
            cls_intersection += intersection.sum(0)
            cls_union += union.sum(0)
            cls_target += target.sum(0)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DIaM Training and Testing Script')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    if args.debug:
        args.test_num = 64
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    # parser = argparse.ArgumentParser(description='DIaM Training and Testing Script')
    # parser.add_argument('--config', type=str, help='Path to YAML config file')
    # parser.add_argument('--split', type=int, help='Data split to use')
    # parser.add_argument('--shot', type=int, help='Number of shots')
    # parser.add_argument('--gpus', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    # args = parser.parse_args()
    # # if args.debug:
    # #     args.test_num = 64
    # #     args.n_runs = 2

    # world_size = len(str(args.gpus))
    # distributed = world_size > 1
    # assert not distributed, 'Testing should not be done in a distributed way'
    # args.distributed = distributed
    # args.port = find_free_port()
    # main_worker(0, world_size, args)
