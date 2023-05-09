# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import unittest
from typing import List, Tuple, Type

import torch
from torch import Tensor, LongTensor

from pangolinn.seq2seq.seq2seq_module_wrapper import PangolinnSeq2SeqModuleWrapper


class BaseTester(unittest.TestCase):
    """
    This class provides basic functions useful for pangolinn sequence-to-sequence testers.
    """
    module_wrapper_class: PangolinnSeq2SeqModuleWrapper.__class__

    def _wrapper_setup(self, pangolinn_class: Type):
        assert self.__class__ is not pangolinn_class, \
            "If you are seeing this error, most likely you have imported " \
            f"{pangolinn_class.__name__}. Instead, import pangolinn.seq2seq and " \
            f"extend seq2seq.{pangolinn_class.__name__}."
        assert self.module_wrapper_class is not None, \
            "Override the class attribute `module_wrapper_class` by setting it to the class of " \
            "your wrapper (e.g., `module_wrapper_class = MyWrapper`)."
        self.module_wrapper: PangolinnSeq2SeqModuleWrapper = self.module_wrapper_class()

    def _forward_with_expected_shape(
            self, x: Tensor, lengths: LongTensor, expected_shape: List[int]) -> Tensor:
        """
        :param x: tensor used as input of the module to be tested
        :param lengths: tensor containing the lengths of each sequence in `x`
        :param expected_shape: list of the expected dimensions
        :return: the output of the forward over the module to be tested
        """
        output = self.module_wrapper.forward(x, lengths)
        self.assertListEqual(
            expected_shape,
            list(output.size()),
            msg=f"Unexpected output shape {output.size()}. Model wrapper should return "
                "a tensor of shape (batch, seq_len, channels), with expected shape "
                f"{expected_shape}.")
        return output

    def _rand_tensor(self, shape: Tuple[int, int, int], dtype: torch.dtype) -> Tensor:
        if dtype.is_floating_point or dtype.is_complex:
            return torch.rand(shape, dtype=dtype)
        else:
            return torch.randint(self.module_wrapper.max_value_allowed, shape, dtype=dtype)
