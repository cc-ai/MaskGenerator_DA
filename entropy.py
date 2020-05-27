from comet_ml import Experiment
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
from pathlib import Path
from torch import nn
# from torch.utils import data
from advent.model.deeplabv2 import get_deeplab_v2
# from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import prob_2_entropy, load_checkpoint_for_evaluation
from torchvision import utils as vutils
import torch.nn.functional as F
import warnings
import numpy as np
from advent.utils.datasets import get_loader
from time import time
from advent.utils.tools import (
    load_opts,
    set_mode,
    # avg_duration,
    flatten_opts,
    print_opts,
    write_images
)
import json
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments 
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")

    # parser.add_argument('--best_iter', type=int, default=30000,
    #                     help='iteration with best mIoU')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='add normalizor to the entropy ranking')
    parser.add_argument('--lambda1', type=float, default=0.67, 
                        help='hyperparameter lambda to split the target domain')
    parser.add_argument('--cfg', type=str, default="shared/advent.yml",
                        help='optional config file')
    parser.add_argument(
        "-d",
        "--data",
        help="yaml file for the data",
        default="shared/config.yml"
    )
    parser.add_argument(
        "-n",
        "--no_check",
        action="store_true",
        default=False,
        help="Prevent sample existence checking for faster dev",
    )
    return parser.parse_args()


def main(args):
    # --------------------------
    # -----  Load Options  -----
    # --------------------------
    args = get_arguments()
    print('Called with args:')
    print(args)
    assert args.cfg is not None, 'Missing cfg file'

    root = Path(__file__).parent.resolve()
    cfg = load_opts(path=root / args.cfg, default="shared/config.yml")
    cfg = set_mode("train", cfg)
    flats = flatten_opts(cfg)
    print_opts(flats)
    cfg.data.loaders.batch_size = 1

    comet_exp = Experiment(workspace=cfg.workspace, project_name=cfg.project_name)
    flats = flatten_opts(cfg)
    comet_exp.log_parameters(flats)
    device = cfg.GPU_ID
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    print('Using config:')
    pprint.pprint(cfg)

    # load model with parameters trained from Inter-domain adaptation
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model_gen {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model_gen = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model_gen.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model_gen.load_state_dict(new_params)
        else:
            model_gen.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    i_iter = cfg.TEST.Model.test_iter
    restore_from = osp.join(cfg.TEST.Model.SNAPSHOT_DIR, f'model_{i_iter}.pth')
    if not osp.exists(restore_from):
        # continue
        if cfg.TEST.WAIT_MODEL:
            print('Waiting for model..!')
            while not osp.exists(restore_from):
                time.sleep(5)
    print("Evaluating model", restore_from)
    load_checkpoint_for_evaluation(model_gen, restore_from, device)

    target_loader = get_loader(cfg, real=True, no_check=args.no_check)
    target_loader_iter = enumerate(target_loader)
    # upsampling layer

    entropy_list = []
    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image = batch['data']['x'][0]
        image = image[None, :, :, :]
        name_image = batch['paths']['x']
        name_mask = batch['paths']['m']
        
        print("Image Shape:", image.shape)
        # if cfg.TEST.Model.fixed_test_size:
        for t in cfg.data.transforms:
            if t == 'resize' and t.ignore:
                interp_target = nn.Upsample(size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
            
            else:
                interp_target = nn.Upsample(size=(cfg.TEST.Model.OUTPUT_SIZE_TARGET[1], cfg.TEST.Model.OUTPUT_SIZE_TARGET[0]), 
                mode='bilinear',align_corners=True)
        if cfg.TEST.Model.save_pred_path_default == True:
            savePathfile = changeSubString(name_mask[0])
            savePath = "/".join(savePathfile.split("/")[:-1]) # get the exact file path without filename
        else:
            savePath = cfg.TEST.Model.save_pred_path_customized
            savePathfile = savePath + "/" + name_image[0].split("/")[-1]
        print(savePath)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        with torch.no_grad():
            try:
                _, pred_trg_main = model_gen(image.cuda(device))
            except Exception:
                print("Clearing CUDA cache!!!")
                torch.cuda.empty_cache()
                _, pred_trg_main = model_gen(image.cuda(device))
            pred_trg_main = interp_target(pred_trg_main)
            output = pred_trg_main.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            # entropy_list.append((name[0], pred_trg_entropy.mean().item() * normalizor))
            
            entropy_list.append((name_image[0], savePathfile, pred_trg_entropy.mean().item()))
            # colorize_save(pred_trg_main, name[0])
            output = torch.tensor(output, dtype=torch.float32)
            output = output[None, :, :]
            output_RGB = output.repeat(3, 1, 1)
            vutils.save_image(output_RGB,savePathfile,normalize=True)
        if cfg.TEST.isShowcomet_ml:
            image = image[0] # change size from [1,x,y,z] to [x,y,z]
            save_images = []

            save_images.append(image)
            # Overlay mask:

            save_fake_mask = (
                image
                - (image * output_RGB)
                + output_RGB
            )
            save_images.append(save_fake_mask)
            save_images.append(output_RGB)

            write_images(
                save_images,
                i_iter,
                comet_exp=comet_exp,
                store_im=cfg.TEST.store_images
            )
    # split the enntropy_list into 
    cluster_subdomain(entropy_list, args.lambda1)


# def cluster_subdomain(entropy_list, lambda1):
#     entropy_list = sorted(entropy_list, key=lambda img: img[2])
#     copy_list = entropy_list.copy()
#     entropy_rank = [(item[0],item[1]) for item in entropy_list]

#     easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
#     hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

#     with open('easy_split.txt','w+') as f:
#         for item in easy_split:
#             f.write('%s\n' % item)

#     with open('hard_split.txt','w+') as f:
#         for item in hard_split:
#             f.write('%s\n' % item)

#     return copy_list


def cluster_subdomain(entropy_list, lambda1):
    entropy_list0 = entropy_list.copy()
    entropy_list0 = sorted(entropy_list0, key=lambda img: img[2])
    entropy_rank = [(item[0],item[1]) for item in entropy_list0]

    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]
    easy_splitDict  = tupleList2DictList(easy_split)
    hard_splitDict  = tupleList2DictList(hard_split)

    # with open('easy_split.txt','w+') as f:
    #     for item in easy_split:
    #         f.write('%s\n' % item)

    # with open('hard_split.txt','w+') as f:
    #     for item in hard_split:
    #         f.write('%s\n' % item)
    
    with open('easy_split.json', 'w', encoding = "utf-8") as outfile:
        json.dump(easy_splitDict, outfile, ensure_ascii=False)
    with open('hard_split.json', 'w', encoding = "utf-8") as outfile:
        json.dump(hard_splitDict, outfile, ensure_ascii=False)
    return


def tupleList2DictList(tuples, keys=['x', 'm']):
    DictList = []
    for Tuple in tuples:
        tmpDict = {}
        for i in range(len(keys)):
            tmpDict[keys[i]] = Tuple[i]
        DictList.append(tmpDict)
    return DictList


def changeSubString(Str, replacing="non_flooded", replaced="realMaskPred"):
    tmp = Str.split(replacing)[0]+replaced+Str.split(replacing)[1]
    return tmp
    
if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    main(args)