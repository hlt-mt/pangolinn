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
from torch import Tensor, LongTensor, nn

from src.pangolinn import padding


class EmbeddingsWrapper(padding.EncoderModuleWrapper):
    """
    Wrapper to test an Embeddings layer which takes integers as input and returns float embeddings.
    """
    def build_encoder_module(self) -> nn.Module:
        return nn.Embedding(self.max_value_allowed, self.num_output_channels, padding_idx=0)

    @property
    def num_input_channels(self) -> int:
        return 1

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.int

    @property
    def num_output_channels(self) -> int:
        return 4

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self.encoder_module(x.squeeze(-1))


class EmbeddingsTestCase(padding.EncoderPaddingTestCase):
    encoder_wrapper_class = EmbeddingsWrapper


if __name__ == '__main__':
    unittest.main()
