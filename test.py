#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp

import cv2
import torch
import torch.nn as nn
import yaml
from addict import Dict
from utils.visualizer import Visualizer
from data.data_loading import *
from models import DeepLabV2_ResNet101_MSC
from utils.metric import label_accuracy_hist, hist_to_score


def load_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    network.load_state_dict(torch.load(save_path))
    print("the network load is in " + save_path)


def save_network(saveDir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = osp.join(saveDir, save_filename)
    torch.save(network.to("cpu").state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()


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
        CONFIG.CROPSIZE,
        phase="crossvali",
        testFlag=True,
        preload=False
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )
    loader_iter = iter(loader)

    # Model
    torch.set_grad_enabled(False)
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    model = nn.DataParallel(model)
    load_network(CONFIG.SAVE_DIR, model, "SateDeepLab", "latest")
    model.to(device)

    #visualizer
    vis = Visualizer(CONFIG.DISPLAYPORT)
    model.eval()

    hist = np.zeros((7, 7))
    for iteration in range(1, 1000):
        print("iteration {}".format(iteration))
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

            for output in outputs:
                # Resize target for {100%, 75%, 50%, Max} outputs
                target_ = resize_target(target, output.size(2))
                target_ = target_.to(device)
                # metric computer
                hist += label_accuracy_hist(target_[0].to("cpu").numpy(), outputs[0].to("cpu").max(0)[1].numpy(), 7)
                # visualizer
                vis.displayImg(inputImgTransBack(data), classToRGB(outputs[0].to("cpu").max(0)[1]),
                                classToRGB(target[0].to("cpu")))
    print("hist is {}".format(hist), file=open("output.txt", "a"))
    _, acc_cls, recall_cls, iu, _ = hist_to_score(hist)
    print("accuracy of every class is {}, recall of every class is {}, iu of every class is {}".format(
        acc_cls, recall_cls, iu), file=open("output.txt", "a"))


if __name__ == "__main__":
    main()
