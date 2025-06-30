# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
from typing import List, Tuple

import datasets
from datasets import DatasetInfo, SplitInfo, get_dataset_config_info, get_dataset_config_names
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs

dataset_names = ["s1k-1.1", "geometry3k"]


def get_flattened_dataset(data_source) -> Tuple[List[datasets.Dataset], List[Tuple[DatasetInfo, SplitInfo]]]:
    """
    Load and flatten the datasets from the specified data source.
    """
    dataset_list = []
    datainfolist = []
    subset_list = get_dataset_config_names(data_source)
    for subset in subset_list:
        data_info = get_dataset_config_info(data_source, subset)
        splits_info = data_info.splits
        for split_name, split_info in splits_info.items():
            if subset in dataset_names:
                dataset = datasets.load_dataset(data_source, subset, split=split_name)
                dataset_list.append(dataset)
                datainfolist.append((data_info, split_info))

    return dataset_list, datainfolist


def get_blank_image():
    """
    Create a blank image to be used as a placeholder.
    """
    blank_image = Image.new("RGB", (28, 28), color=(255, 255, 255))
    return blank_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/open_mm_recipe_image")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", default="luodian/think-in-modality", help="The data source to load the dataset from")

    args = parser.parse_args()

    data_source = args.data_source
    dataset_list, datainfolist = get_flattened_dataset(data_source)
    dataset = datasets.concatenate_datasets(dataset_list)

    train_dataset = dataset

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")
            if len(images) > 0 and "<image>" not in prompt:
                prompt = "<image>" * len(images) + prompt
            elif len(images) == 0:
                images = [get_blank_image()]
                prompt = "<image>" + prompt

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
