# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
from tqdm import tqdm

import monai.transforms as mt
from monai.apps.datasets import DecathlonDataset
from monai.apps.utils import download_and_extract
from monai.utils.enums import PostFix

class Slicer(mt.MapTransform):
    @staticmethod
    def get_num_non_zero_voxels(im):
        """Gets the number of non zero voxels along the last dimension. If input image
        has 100 voxels in z-dimension, then returned will be a 100-element array with number
        of non-zero voxels in that dimension."""
        return (im > 0).reshape(-1, im.shape[-1]).sum(0)

    def get_slice(self, _):
        raise NotImplementedError()

    def __call__(self, data):
        _slice = self.get_slice(data)
        cropper = mt.SpatialCropd(self.keys, roi_slices=[slice(None), slice(None), slice(_slice,_slice+1)])
        out = cropper(data)
        for k in self.keys:
            out[k] = out[k].squeeze(-1)
        return out


class SliceWithMaxNumLabelsd(Slicer):
    """Get a 2D slice of a 3D volume with the maximum number of non-zero voxels in the label."""

    def __init__(self, keys, label_key="label"):
        self.keys = keys
        self.label_key = label_key

    def get_slice(self, data):
        num_non_zero_lbl = self.get_num_non_zero_voxels(data[self.label_key])
        return num_non_zero_lbl.argmax(0)


class SliceWithNoLabelsd(Slicer):
    """
    Get a 2D slice of a 3D volume that contains only background class (0) in the segmentation.
    Of all the 2D slices that contain only zeros in the segmentation, pick the slice that has the
    largest number of non-zero voxels in the image. This ensures that we have an image that contains
    lots of healthy tissue.
    """
    def __init__(self, keys, label_key="label", image_key="image"):
        self.keys = keys
        self.label_key = label_key
        self.image_key = image_key

    def get_slice(self, data):
        num_non_zero_img = self.get_num_non_zero_voxels(data[self.image_key])
        num_non_zero_lbl = self.get_num_non_zero_voxels(data[self.label_key])
        # ignore any points where the label is non-zero
        num_non_zero_img[num_non_zero_lbl != 0] = 0
        # get slice with most non-zero voxels
        return num_non_zero_img.argmax(0)


def download_data(task, download_path):
    """Download data (if necessary) and return a list of images and corresponding labels."""
    resource, md5 = DecathlonDataset.resource[task], DecathlonDataset.md5[task]
    compressed_file = os.path.join(download_path, task + ".tar")
    data_dir = os.path.join(download_path, task)
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, download_path, hash_val=md5)
    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image, "label": label} for image, label in zip(images, labels)]
    return data_dicts


def main(task, path, download_path):
    data_dicts = download_data(task, download_path)

    # list of transforms to convert to 2d slice
    transform_2d_slice = mt.Compose(
        [
            mt.LoadImaged(["image", "label"]),
            mt.AsChannelFirstd("image"),
            mt.AddChanneld("label"),
            mt.CopyItemsd(
                ["image", "label", PostFix.meta("image"), PostFix.meta("label")],
                times=2,
                names=["image_tumour", "label_tumour", PostFix.meta("image_tumour"), PostFix.meta("label_tumour"),
                       "image_healthy", "label_healthy", PostFix.meta("image_healthy"), PostFix.meta("label_healthy")],
            ),
            SliceWithMaxNumLabelsd(["image_tumour", "label_tumour"]),
            SliceWithNoLabelsd(["image_healthy"]),
            mt.SaveImaged("image_tumour", output_dir=os.path.join(path, "image_tumour"), output_postfix="", separate_folder=False, resample=False, print_log=False),
            mt.SaveImaged("label_tumour", output_dir=os.path.join(path, "label_tumour"), output_postfix="", separate_folder=False, resample=False, print_log=False),
            mt.SaveImaged("image_healthy", output_dir=os.path.join(path, "image_healthy"), output_postfix="", separate_folder=False, resample=False, print_log=False),
        ]
    )

    for data in tqdm(data_dicts):
        # skip the 2d extraction if possible
        if len(glob(os.path.join(path, "*", os.path.basename(data["image"])))) == 3:
            continue
        # extract the slice
        _ = transform_2d_slice(data)


def print_input_args(args):
    data = dict(sorted(args.items()))
    col_width = max(len(i) for i in data.keys())
    for k, v in data.items():
        print(f"\t{k:<{col_width}}: {v if v is not None else 'None'}")


if __name__ == "__main__":
    default_download_root_dir = os.environ.get("MONAI_DATA_DIRECTORY")
    if default_download_root_dir is None:
        default_download_root_dir = tempfile.mkdtemp()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add arguments
    parser.add_argument(
        "-t",
        "--task",
        help="Task to generate the 2d dataset from.",
        type=str,
        choices=["Task01_BrainTumour"],
        default="Task01_BrainTumour",
    )
    parser.add_argument(
        "-d", "--download_path", help="Path for downloading full dataset.", type=str, default=default_download_root_dir
    )
    parser.add_argument("-p", "--path", help="Path for output. Default: download_path/{task}2D", type=str)

    # parse input arguments
    args = vars(parser.parse_args())

    # set default output path if necessary
    if args["path"] is None:
        args["path"] = os.path.join(args["download_path"], args["task"] + "2D")

    # print args and run the 2d extraction
    print_input_args(args)
    main(**args)
