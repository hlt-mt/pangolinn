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

import torch
from torch import Tensor, LongTensor, nn, BoolTensor

from src.pangolinn import padding


class LinearPaddingSafeWrapper(padding.EncoderModuleWrapper):
    """
    Wrapper to test a linear layer which does properly handles padding.
    """
    def build_encoder_module(self) -> nn.Module:
        return nn.Linear(self.num_input_channels, self.num_output_channels)

    @property
    def num_input_channels(self) -> int:
        return 4

    @staticmethod
    def padding_mask_from_lens(lengths: LongTensor) -> BoolTensor:
        bsz = lengths.size(0)
        max_len = lengths.max()
        padding_mask = torch.arange(max_len).unsqueeze(0).expand(bsz, -1)
        return padding_mask >= lengths.unsqueeze(1).expand(-1, max_len)

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        output_padding_unsafe = self.encoder_module(x)
        padding_mask = self.padding_mask_from_lens(lengths)
        return output_padding_unsafe.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class LinearPaddingSafeTestCase(padding.EncoderPaddingTestCase):
    encoder_wrapper_class = LinearPaddingSafeWrapper


if __name__ == '__main__':
    unittest.main()
