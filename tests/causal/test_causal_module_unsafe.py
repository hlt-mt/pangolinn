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

from torch import Tensor, LongTensor, nn

from pangolinn import seq2seq


class TransformerDecoderWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    """
    Wrapper to test a layer that does not look at the future, so it is safe in causal models.
    """
    def build_module(self) -> nn.Module:
        return nn.TransformerEncoderLayer(
            self.num_input_channels, 1, dim_feedforward=8, batch_first=True)

    @property
    def num_input_channels(self) -> int:
        return 4

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x)


class NonCausalModuleTestCase(seq2seq.CausalTestCase):
    module_wrapper_class = TransformerDecoderWrapper

    def test_not_looking_at_the_future(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_not_looking_at_the_future()
        self.assertIn("Tensor-likes are not close", str(ae.exception))

    def test_gradient_not_flowing_from_future(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_gradient_not_flowing_from_future()
        self.assertIn("within 7 places", str(ae.exception))


if __name__ == '__main__':
    unittest.main()
