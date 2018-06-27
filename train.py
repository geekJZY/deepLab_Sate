#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tqdm import tqdm
from utils.visualizer import Visualizer
from data.data_loading import *
from models import DeepLabV2_ResNet101_MSC
from utils.loss import CrossEntropyLoss2d


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


def get_lr_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    optimizer.param_groups[1]["lr"] = 10 * new_lr
    optimizer.param_groups[2]["lr"] = 20 * new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()


def main():
    config = "config/cocostuff.yaml"
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))
    CONFIG.LOGNAME = osp.join(CONFIG.SAVE_DIR, "log.txt")

    # Dataset
    dataset = MultiDataSet(
        CONFIG.ROOT,
        CONFIG.CROPSIZE
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    state_dict = torch.load(CONFIG.INIT_MODEL)
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    model = nn.DataParallel(model)
    # read old version
    if CONFIG.ITER_START != 1:
        load_network(CONFIG.SAVE_DIR, model, "SateDeepLab", "latest")
        print("load previous model succeed, training start from iteration {}".format(CONFIG.ITER_START))
    model.to(device)

    # Optimizer
    optimizer = {
        "sgd": torch.optim.SGD(
            # cf lr_mult and decay_mult in train.prototxt
            params=[
                {
                    "params": get_lr_params(model.module, key="1x"),
                    "lr": CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_lr_params(model.module, key="10x"),
                    "lr": 10 * CONFIG.LR,
                    "weight_decay": CONFIG.WEIGHT_DECAY,
                },
                {
                    "params": get_lr_params(model.module, key="20x"),
                    "lr": 20 * CONFIG.LR,
                    "weight_decay": 0.0,
                },
            ],
            momentum=CONFIG.MOMENTUM,
        )
    }.get(CONFIG.OPTIMIZER)

    # Loss definition
    criterion = CrossEntropyLoss2d()
    criterion.to(device)

    #visualizer
    vis = Visualizer(CONFIG.DISPLAYPORT)

    model.train()
    model.module.scale.freeze_bn()

    for iteration in range(CONFIG.ITER_START, CONFIG.ITER_MAX + 1):

        # Set a learning rate
        poly_lr_scheduler(
            optimizer=optimizer,
            init_lr=CONFIG.LR,
            iter=iteration - 1,
            lr_decay_iter=CONFIG.LR_DECAY,
            max_iter=CONFIG.ITER_MAX,
            power=CONFIG.POLY_POWER,
        )

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG.ITER_SIZE + 1):
            try:
                data, target = next(loader_iter)
            except:
                loader_iter = iter(loader)
                data, target = next(loader_iter)

            # Image
            data = data.to(device)

            # Propagate forward
            outputs = model(data)

            # Loss
            loss = 0
            for output in outputs:
                # Resize target for {100%, 75%, 50%, Max} outputs
                target_ = resize_target(target, output.size(2))
                target_ = target_.to(device)
                # Compute crossentropy loss
                loss += criterion(output, target_)

            # Backpropagate (just compute gradients wrt the loss)
            loss /= float(CONFIG.ITER_SIZE)
            loss.backward()

            iter_loss += float(loss)

        # Update weights with accumulated gradients
        optimizer.step()
        # Visualizer and Summery Writer
        if iteration % CONFIG.ITER_TF == 0:
            print("itr {}, loss is {}".format(iteration, iter_loss), file=open(CONFIG.LOGNAME, "a"))
            # vis.drawLine(torch.FloatTensor([iteration]), torch.FloatTensor([iter_loss]))
            # vis.displayImg(inputImgTransBack(data), classToRGB(outputs[3][0].to("cpu").max(0)[1]),
            #                classToRGB(target[0].to("cpu")))

        # Save a model
        if iteration % CONFIG.ITER_SNAP == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateDeepLab", iteration)

        # Save a model
        if iteration % 100 == 0:
            save_network(CONFIG.SAVE_DIR, model, "SateDeepLab", "latest")

    save_network(CONFIG.SAVE_DIR, model, "SateDeepLab", "final")


if __name__ == "__main__":
    main()
