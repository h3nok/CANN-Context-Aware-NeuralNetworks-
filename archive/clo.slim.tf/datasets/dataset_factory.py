# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import caltech101
from datasets import caltech256
from datasets import cifar10
from datasets import cifar100
from datasets import flowers
from datasets import imagenet

datasets_map = {
    # 'cats_vs_dogs':cats_vs_dogs,
    # 'flowers':flowers,
    'cifar10': cifar10,
    'caltech101': caltech101,
    'caltech256': caltech256,
    'cifar100': cifar100,
    'imagenet': imagenet,
    'flowers': flowers
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a pipe name and a split_name returns a Dataset.

  Args:
    name: String, the name of the pipe.
    split_name: A train/test split name.
    dataset_dir: The directory where the pipe files are stored.
    file_pattern: The file pattern to use for matching the pipe source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each pipe is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the pipe `name` is unknown.clear
  """
    if name not in datasets_map:
        raise ValueError('Name of pipe unknown %s' % name)

    return datasets_map[name].get_split(
        split_name,
        dataset_dir,
        file_pattern,
        reader)
