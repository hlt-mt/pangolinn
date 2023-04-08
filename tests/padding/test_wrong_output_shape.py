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

from src.pangolinn import padding


class LinearPaddingWrongShapeWrapper(padding.EncoderModuleWrapper):
    """
    Wrapper to test a linear layer which does not return the shape expected
    according to `num_output_channels`.
    """
    def build_encoder_module(self) -> nn.Module:
        return nn.Linear(self.num_input_channels, self.num_input_channels)

    @property
    def num_input_channels(self) -> int:
        return 4

    @property
    def num_output_channels(self) -> int:
        return 2

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self.encoder_module(x)


class LinearPaddingWrongShapeTestCase(padding.EncoderPaddingTestCase):
    encoder_wrapper_class = LinearPaddingWrongShapeWrapper

    def test_padding_area_is_zero(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_padding_area_is_zero()
        self.assertIn("Unexpected output shape", str(ae.exception))

    def test_batch_size_does_not_matter(self):
        with self.assertRaises(AssertionError) as ae:
            super().test_batch_size_does_not_matter()
        self.assertIn("Unexpected output shape", str(ae.exception))


if __name__ == '__main__':
    unittest.main()
