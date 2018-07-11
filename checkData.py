#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os.path as osp

import cv2
import torch
import yaml
from addict import Dict
from data.data_loading import *
import numpy as np


def main():
    config = "config/cocostuff.yaml"

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

    print("Dataset is loaded")

    list_95_100 = np.zeros(7, dtype=np.float32)
    list_85_95 = np.zeros(7, dtype=np.float32)
    list_70_85 = np.zeros(7, dtype=np.float32)
    list_0_70 = np.zeros(7, dtype=np.float32)
    cnt_0_70 = 0

    for itr in range(100):
        print("iteration {}".format(itr))
        for index, (_, label) in enumerate(dataset):
            print("calculate img{}".format(index))
            hist = np.bincount(label.flatten(), minlength=7)
            hist = hist / np.sum(hist)
            if np.nanmax(hist) > 0.95:
                list_95_100[np.argmax(hist)] += 1
            elif np.nanmax(hist) > 0.85:
                list_85_95[np.argmax(hist)] += 1
            elif np.nanmax(hist) > 0.70:
                list_70_85[np.argmax(hist)] += 1
            else:
                cnt_0_70 += 1
                list_0_70 += hist
        dataDistribution = list_0_70 / np.sum(list_0_70)
        print("iteration {}".format(itr), file=open("statistics.txt", "a"))
        print("list_95_100 is {}".format(list_95_100), file=open("statistics.txt", "a"))
        print("list_85_95 is {}".format(list_85_95), file=open("statistics.txt", "a"))
        print("list_70_85 is {}".format(list_70_85), file=open("statistics.txt", "a"))
        print("cnt_0_70 is {}".format(cnt_0_70), file=open("statistics.txt", "a"))
        print("dataDistribution is {}".format(dataDistribution), file=open("statistics.txt", "a"))


if __name__ == "__main__":
    main()