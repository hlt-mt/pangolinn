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


class LinearPaddingUnsafeWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    """
    Wrapper to test a linear layer which does not handle padding properly.
    It does not return zeroes in the padding area, although the linear layer
    returns the same output in the non-padding area.
    """
    def build_module(self) -> nn.Module:
        return nn.Linear(self.num_input_channels, self.num_output_channels)

    @property
    def num_input_channels(self) -> int:
        return 4

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x)


class LinearPaddingUnsafeTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = LinearPaddingUnsafeWrapper

    def test_padding_area_is_zero(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_padding_area_is_zero()
        self.assertIn("non-zero entries in", str(ae.exception))


if __name__ == '__main__':
    unittest.main()
