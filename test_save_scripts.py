import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load

from advent.utils.tools import (
    print_losses,
    tesnorDict2numDict,
    write_images
)


def eval_best(cfg, model,
              device, test_loader, interp, comet_exp,
              fixed_test_size, verbose):
    cur_best_miou = -1
    cur_best_model = ''
    i_iter = cfg.TEST.Model.test_iter
    restore_from = osp.join(cfg.TEST.Model.SNAPSHOT_DIR, f'model_{i_iter}.pth')
    if not osp.exists(restore_from):
        # continue
        if cfg.TEST.WAIT_MODEL:
            print('Waiting for model..!')
            while not osp.exists(restore_from):
                time.sleep(5)
    print("Evaluating model", restore_from)
    load_checkpoint_for_evaluation(model, restore_from, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = enumerate(test_loader)
    for index in tqdm(range(len(test_loader))):
        _, batch = test_iter.__next__()
        # print(test_iter.__next__())
        image, label = batch['data']['x'][0], batch['data']['m'][0]
        image = image[None, :, :, :]
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            pred_main = model(image.cuda(device))[1]
            output = interp(pred_main).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label0 = label.numpy()[0]
        hist += fast_hist(label0.flatten(), output.flatten(), cfg.NUM_CLASSES)
        output = torch.tensor(output, dtype=torch.float32)
        output = output[None, :, :]
        output_RGB = output.repeat(3, 1, 1)
    
        inters_over_union_classes = per_class_iu(hist)
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = f'model_{i_iter}.pth'
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        mious = {'Current mIoU': computed_miou,
                    'Current best model': cur_best_model,
                    'Current best mIoU': cur_best_miou}
        comet_exp.log_metrics(mious)
        image = image[0] # change size from [1,x,y,z] to [x,y,z]
        save_images = []

        save_images.append(image)
        # Overlay mask:

        save_mask = (
            image
            - (image * label.repeat(3, 1, 1))
            + label.repeat(3, 1, 1)
        )

        save_fake_mask = (
            image
            - (image * output_RGB)
            + output_RGB
        )
        save_images.append(save_mask)
        save_images.append(save_fake_mask)
        save_images.append(label.repeat(3, 1, 1))
        save_images.append(output_RGB)

        write_images(
            save_images,
            i_iter,
            comet_exp=comet_exp,
            store_im=cfg.TEST.store_images
        )
    return computed_miou, cur_best_model, cur_best_miou


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))