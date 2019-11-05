import numpy as np
from PIL import Image as PImage
import torch
import os
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from losses import ContrastiveLoss, OnlineContrastiveLoss
from utils import HardNegativePairSelector

from fastai.vision import *


def pad_to_size(img: torch.Tensor, size):
    padded = torch.zeros(img.shape[0], size[1], size[0])
    end1, end2 = min(img.shape[1], size[1]), min(img.shape[2], size[0])
    padded[:, :end1, :end2] = img[:, :end1, :end2]
    return padded


class ImageDataset(Dataset):
    def __init__(
        self, path, classes, tfms=None, grayscale=False, check_integrity=False
    ):
        super().__init__()
        self._data = []
        self.path = path
        self.to_tensor = transforms.ToTensor()
        self.classes = classes
        self.c = len(self.classes)
        for label in self.classes:
            for imgfile in os.listdir(os.path.join(path, label)):
                try:
                    if check_integrity:
                        img = PImage.open(os.path.join(path, label, imgfile))
                        if grayscale:
                            img = img.convert("L")
                        else:
                            img = img.convert("RGB")
                        if tfms is not None:
                            tfms(img)
                        del img

                    self._data.append(
                        (os.path.join(path, label, imgfile), self.classes.index(label))
                    )

                except RuntimeError as e:
                    print(e)
                    continue

        self._tfms = tfms

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        imgpath, label = self._data[idx]
        img = PImage.open(imgpath)
        img = img.convert("RGB")
        if self._tfms is not None:
            img = self._tfms(img)
        return img, label  # .index(label)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

            random_state = np.random.RandomState(29)

            positive_pairs = [
                [
                    i,
                    random_state.choice(
                        self.label_to_indices[self.test_labels[i].item()]
                    ),
                    1,
                ]
                for i in range(0, len(self.test_data), 2)
            ]

            negative_pairs = [
                [
                    i,
                    random_state.choice(
                        self.label_to_indices[
                            np.random.choice(
                                list(
                                    self.labels_set - set([self.test_labels[i].item()])
                                )
                            )
                        ]
                    ),
                    0,
                ]
                for i in range(1, len(self.test_data), 2)
            ]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = PImage.fromarray(img1.numpy(), mode="L")
        img2 = PImage.fromarray(img2.numpy(), mode="L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class SiameseImage(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, widgets_dataset: ImageDataset, train, margin=1.0):
        self.widgets_dataset = widgets_dataset
        self.classes = ["negative", "positive"]
        self.c = 2
        self.loss_func = ContrastiveLoss(margin)
        self.path = widgets_dataset.path

        self.train = train
        self.transform = self.widgets_dataset._tfms
        self._data_paths, self._labels = list(zip(*self.widgets_dataset._data))
        self._labels = torch.LongTensor(self._labels)
        self.labels_set = set(self._labels.numpy())
        self.label_to_indices = {
            label: np.where(self._labels.numpy() == label)[0]
            for label in self.labels_set
        }

        if not self.train:
            random_state = np.random.RandomState(29)

            positive_pairs = [
                [
                    i,
                    random_state.choice(self.label_to_indices[self._labels[i].item()]),
                    1,
                ]
                for i in range(0, len(self._data_paths), 2)
            ]

            negative_pairs = [
                [
                    i,
                    random_state.choice(
                        self.label_to_indices[
                            np.random.choice(
                                list(self.labels_set - set([self._labels[i].item()]))
                            )
                        ]
                    ),
                    0,
                ]
                for i in range(1, len(self._data_paths), 2)
            ]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            imgpath1 = self._data_paths[index]
            label1 = self._labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            imgpath2 = self._data_paths[siamese_index]
        else:
            imgpath1 = self._data_paths[self.test_pairs[index][0]]
            imgpath2 = self._data_paths[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        return (imgpath1, imgpath2), target

    def __len__(self):
        return len(self.widgets_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

            random_state = np.random.RandomState(29)

            triplets = [
                [
                    i,
                    random_state.choice(
                        self.label_to_indices[self.test_labels[i].item()]
                    ),
                    random_state.choice(
                        self.label_to_indices[
                            np.random.choice(
                                list(
                                    self.labels_set - set([self.test_labels[i].item()])
                                )
                            )
                        ]
                    ),
                ]
                for i in range(len(self.test_data))
            ]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = PImage.fromarray(img1.numpy(), mode="L")
        img2 = PImage.fromarray(img2.numpy(), mode="L")
        img3 = PImage.fromarray(img3.numpy(), mode="L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class COCODataset(Dataset):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(
        1, 1, 3
    )
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(
        1, 1, 3
    )

    def __init__(self, opt, split):
        super().__init__()
        self.data_dir = os.path.join(opt.data_dir, "coco")
        self.img_dir = os.path.join(self.data_dir, "{}2017".format(split))
        if split == "test":
            self.annot_path = os.path.join(
                self.data_dir, "annotations", "image_info_test-dev2017.json"
            ).format(split)
        else:
            if opt.task == "exdet":
                self.annot_path = os.path.join(
                    self.data_dir, "annotations", "instances_extreme_{}2017.json"
                ).format(split)
            else:
                self.annot_path = os.path.join(
                    self.data_dir, "annotations", "instances_{}2017.json"
                ).format(split)
        self.max_objs = 128
        self.class_name = [
            "__background__",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        self._valid_ids = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [
            (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
            for v in range(1, self.num_classes + 1)
        ]
        self._data_rng = np.random.RandomState(1337)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array(
            [
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938],
            ],
            dtype=np.float32,
        )
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print("==> initializing coco 2017 {} data.".format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(
            self.convert_eval_format(results),
            open("{}/results.json".format(save_dir), "w"),
        )

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes("{}/results.json".format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1, img2
        self.obj, self.data = (
            (img1, img2),
            [pad_to_size(img1.data, (512, 512)), pad_to_size(img2.data, (512, 512))],
        )

    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, **kwargs)
        self.data = [
            pad_to_size(self.img1.data, (512, 512)),
            pad_to_size(self.img2.data, (512, 512)),
        ]
        return self

    def to_one(self):
        return Image(torch.cat(self.data, 2))

    def __repr__(self):
        return f"({self.img1}, {self.img2})"


class SiameseImageList(ImageList):
    _label_cls = CategoryList

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    @classmethod
    def from_datasets(cls, dataset_train, dataset_val, **kwargs):
        items_train = [dataset_train[i] for i in range(len(dataset_train))]
        items_val = [dataset_val[i] for i in range(len(dataset_val))]

        il_train = cls(items_train)
        il_val = cls(items_val)

        ils = ItemLists(os.path.join(dataset_train.path, "../"), il_train, il_val)

        return ils.label_from_func(lambda item: item[1])

    def get(self, i):
        imgs, target = self.items[i]
        img1, img2 = imgs
        return ImageTuple(self.open(img1), self.open(img2))

    def reconstruct(self, t: Tensor):
        return ImageTuple(Image(t[0]), Image(t[1]))

    def show_xys(self, xs, ys, figsize: Tuple[int, int] = (12, 6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows, rows, figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int] = None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = ifnone(figsize, (12, 3 * len(xs)))
        fig, axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle("Ground truth / Predictions", weight="bold", size=14)
        for i, (x, z) in enumerate(zip(xs, zs)):
            x.to_one().show(ax=axs[i, 0], **kwargs)
            z.to_one().show(ax=axs[i, 1], **kwargs)


class ImageEmbedList(ImageList):
    def __init__(
        self, *args, convert_mode="RGB", after_open: Callable = None, **kwargs
    ):
        super().__init__(
            *args, convert_mode=convert_mode, after_open=after_open, **kwargs
        )
        self.label_cls = PseudoCategoryList


class PseudoCategoryList(CategoryList):
    def __init__(
        self,
        items: Iterator,
        classes: Collection = None,
        label_delim: str = None,
        **kwargs,
    ):
        super().__init__(items, classes=classes, **kwargs)
        self.loss_func = OnlineContrastiveLoss(1.0, HardNegativePairSelector())

    def analyze_pred(self, pred):
        return pred

    def reconstruct(self, t):
        if isinstance(t, int) or len(t.size()) == 0:
            return Category(t, self.classes[t])
        else:
            return EmptyLabel()
