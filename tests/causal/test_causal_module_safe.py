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

from pangolinn import seq2seq


class TransformerDecoderWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    """
    Wrapper to test a layer that does not look at the future, so it is safe in causal models.
    """
    def build_module(self) -> nn.Module:
        return nn.TransformerDecoderLayer(
            self.num_input_channels, 1, dim_feedforward=8, batch_first=True)

    @property
    def num_input_channels(self) -> int:
        return 4

    @staticmethod
    def generate_attention_mask(sz: int, device: str = "cpu") -> torch.Tensor:
        """ Generate the attention mask for causal decoding """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
        mask.masked_fill_(mask == 0, float("-inf"))
        mask.masked_fill_(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        fake_encoder_out = torch.ones(x.shape[0], 1, self.num_input_channels)
        fake_encoder_out = fake_encoder_out.to(x.device)
        tgt_mask = self.generate_attention_mask(x.shape[1])
        return self._module(x, memory=fake_encoder_out, tgt_mask=tgt_mask)


class CausalDecoderTestCase(seq2seq.CausalTestCase):
    module_wrapper_class = TransformerDecoderWrapper


if __name__ == '__main__':
    unittest.main()
