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


class ConvPaddingUnsafeWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    """
    Wrapper to test a sequence of two convolutional layers that do not
    consider padding and, as such, (wrongly) return different outputs
    according to the padding amount.
    """
    def build_module(self) -> nn.Module:
        return nn.Sequential(self.build_conv1d(), self.build_conv1d())

    def build_conv1d(self) -> nn.Module:
        return nn.Conv1d(
            self.num_input_channels, self.num_output_channels, kernel_size=3, stride=2, padding=1)

    @property
    def num_input_channels(self) -> int:
        return 4

    @property
    def sequence_downsampling_factor(self) -> int:
        return 4

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x.transpose(1, 2)).transpose(1, 2)


class ConvPaddingUnsafeTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = ConvPaddingUnsafeWrapper

    def test_padding_area_is_zero(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_padding_area_is_zero()
        self.assertIn("non-zero entries in", str(ae.exception))

    def test_batch_size_does_not_matter(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_batch_size_does_not_matter()
        self.assertIn("Tensor-likes are not close", str(ae.exception))


if __name__ == '__main__':
    unittest.main()
