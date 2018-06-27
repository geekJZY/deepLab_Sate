# Prepare Dataset
from os import listdir
from os.path import join
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
import cv2
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    indices = np.where(np.all(label == (0, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 1
    indices = np.where(np.all(label == (255, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (255, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 3
    indices = np.where(np.all(label == (0, 255, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 4
    indices = np.where(np.all(label == (0, 0, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 5
    indices = np.where(np.all(label == (255, 255, 255), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 6
    indices = np.where(np.all(label == (0, 0, 0), axis=-1))
    classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3))
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 255]
    indices = np.where(label == 2)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 0]
    indices = np.where(label == 3)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 255]
    indices = np.where(label == 4)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 255, 0]
    indices = np.where(label == 5)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 255]
    indices = np.where(label == 6)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap.astype(np.uint8))


def inputImgTransBack(inputs):
    image = inputs[0].to("cpu")
    image[0] = image[0] + 0.3964
    image[1] = image[1] + 0.3695
    image[2] = image[2] + 0.2726
    return image


class MultiDataSet(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, cropSize, phase="train", labelExtension='.png', testFlag=False):
        super(MultiDataSet, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.cropSize = cropSize
        self.mean = np.array([0.3964, 0.3695, 0.2726])
        self.fileDir = join(self.root, phase)
        self.labelExtension = labelExtension
        self.testFlag = testFlag
        self.image_filenames = [image_name for image_name in listdir(self.fileDir + '/Sat') if
                                is_image_file(image_name)]
        self.classdict = {1: "urban", 2: "agriculture", 3: "rangeland", 4: "forest", 5: "water", 6: "barren",
                          0: "unknown"}

    def __getitem__(self, index):
        Satsample = cv2.imread(join(self.fileDir, 'Sat/' + self.image_filenames[index]))
        image = cv2.cvtColor(Satsample, cv2.COLOR_BGR2RGB)
        labelsamplename = find_label_map_name(self.image_filenames[index], self.labelExtension)
        labelsample = cv2.imread(join(self.fileDir, 'Label/' + labelsamplename))
        label = cv2.cvtColor(labelsample, cv2.COLOR_BGR2RGB)
        label = RGB_mapping_to_class(label)
        image, label = self._transform(image, label)

        image = image/255 - self.mean
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int64)

    def _transform(self, image, label):
        # Scaling
        scale_factor = random.uniform(1, 2)
        scale = math.ceil(scale_factor * self.cropSize)

        h, w, _ = image.shape
        w_offset = random.randint(0, max(0, w - scale - 1))
        h_offset = random.randint(0, max(0, h - scale - 1))

        image = image[h_offset:h_offset + scale,
                w_offset:w_offset + scale, :]
        label = label[h_offset:h_offset + scale,
                w_offset:w_offset + scale]
        image = cv2.resize(
            image,
            (self.cropSize, self.cropSize),
            interpolation=cv2.INTER_LINEAR,
        )
        label = cv2.resize(
            label,
            (self.cropSize, self.cropSize),
            interpolation=cv2.INTER_NEAREST,
        )

        if not self.testFlag:
            # Rotate
            rotate_time = np.random.randint(low=0, high=4)
            np.rot90(image, rotate_time)
            np.rot90(label, rotate_time)

            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label

    def __len__(self):
        return len(self.image_filenames)

# dataset_train = MultiDataSet("/home/ckx9411sx/deepGlobe/land-train")
# np.save('/home/ckx9411sx/deepGlobe/temp/train_data_mc.npy', dataset_train)
# print(dataset_train)
