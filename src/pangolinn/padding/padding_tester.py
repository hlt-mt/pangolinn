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
from typing import List, Tuple

import torch
from torch import Tensor, LongTensor

from src.pangolinn.padding.encoder_module_wrapper import EncoderModuleWrapper


class EncoderPaddingTestCase(unittest.TestCase):
    """
    This class provides unit tests to enforce that the module to be tested properly handles
    padding, i.e. returns the same results regardless of the amount of padding present in
    the batched input sequences.

    To use it to test your network:

     1. create a `EncoderModuleWrapper` that wraps your module (e.g. `MyWrapper`);
     2. create test class that extends `EncoderPaddingTestCase`;
     3. in your test class, override the class attribute `encoder_wrapper_class` by setting it to
        the **class** of your wrapper (e.g., `encoder_wrapper_class = MyWrapper`);
    """
    encoder_wrapper_class: EncoderModuleWrapper.__class__

    def setUp(self) -> None:
        assert self.__class__ is not EncoderPaddingTestCase,  \
            "If you are seeing this error, most likely you have imported " \
            "EncoderPaddingTestCase. Instead, import pangolinn.padding and " \
            "extend padding.EncoderPaddingTestCase."
        assert self.encoder_wrapper_class is not None, \
            "Override the class attribute `encoder_wrapper_class` by setting it to the class of " \
            "your wrapper (e.g., `encoder_wrapper_class = MyWrapper`)."
        self.model_wrapper: EncoderModuleWrapper = self.encoder_wrapper_class()

    def __forward_with_expected_shape(
            self, x: Tensor, lengths: LongTensor, expected_shape: List[int]) -> Tensor:
        """
        :param x: tensor used as input of the module to be tested
        :param lengths: tensor containing the lengths of each sequence in `x`
        :param expected_shape: list of the expected dimensions
        :return: the output of the forward over the module to be tested
        """
        output = self.model_wrapper.forward(x, lengths)
        self.assertListEqual(
            expected_shape,
            list(output.size()),
            msg=f"Unexpected output shape {output.size()}. Model wrapper should return "
                "a tensor of shape (batch, seq_len, channels), with expected shape "
                f"{expected_shape}.")
        return output

    def __rand_tensor(self, shape: Tuple[int, int, int], dtype: torch.dtype) -> Tensor:
        if dtype.is_floating_point or dtype.is_complex:
            return torch.rand(shape, dtype=dtype)
        else:
            return torch.randint(self.model_wrapper.max_value_allowed, shape, dtype=dtype)

    def test_padding_area_is_zero(self):
        """
        Tests that the padding area of the output contains all zeroes.
        Although the presence of non-zero elements in the passing area is not an issue
        on its own, elaborations (e.g., convolutions) on top of non-zero-padded tensors
        might cause issues.
        """
        # test both with multipliers of 4 and not, as many systems
        # may have a 2x or 4x downsampling factor, so we test both when
        # seq_len is divisible or not by the downsampling factor
        for max_batch_seq_len, shorter_seq_len in [(27, 13), (24, 16)]:
            rand_batch = self.__rand_tensor(
                (2, max_batch_seq_len, self.model_wrapper.num_input_channels),
                self.model_wrapper.input_dtype)
            rand_batch[1, shorter_seq_len:, :] = 0
            batch_lens = torch.LongTensor([max_batch_seq_len, shorter_seq_len])
            expected_shape = [
                2,
                self.model_wrapper.output_sequence_length(max_batch_seq_len),
                self.model_wrapper.num_output_channels]
            output = self.__forward_with_expected_shape(rand_batch, batch_lens, expected_shape)
            expected_shorter_seq_len = self.model_wrapper.output_sequence_length(shorter_seq_len)
            padding_area = output[1, expected_shorter_seq_len:, :]
            self.assertGreater(padding_area.numel(), 0)
            self.assertTrue(torch.all(padding_area == 0), f"non-zero entries in {padding_area}")

    def test_batch_size_does_not_matter(self):
        """
        Tests that for the same input we get the same output regardless of the amount of padding.
        """
        for max_batch_seq_len, shorter_seq_len in [(27, 13), (24, 16)]:
            rand_batch = self.__rand_tensor(
                (5, max_batch_seq_len, self.model_wrapper.num_input_channels),
                self.model_wrapper.input_dtype)
            # multiple padded elements of same len
            for b_idx in [1, 2, 3]:
                rand_batch[b_idx, shorter_seq_len:, :] = 0
            # sequence of len 1
            rand_batch[4, 1:, :] = 0
            batch_lens = torch.LongTensor([
                max_batch_seq_len, shorter_seq_len, shorter_seq_len, shorter_seq_len, 1])
            expected_shape = [
                5,
                self.model_wrapper.output_sequence_length(max_batch_seq_len),
                self.model_wrapper.num_output_channels]
            output = self.__forward_with_expected_shape(rand_batch, batch_lens, expected_shape)
            for i in range(5):
                item_len = batch_lens[i].item()
                item_valid_tokens = rand_batch[i, :item_len, :].unsqueeze(0)
                output_wo_padding = self.model_wrapper.forward(
                    item_valid_tokens, LongTensor([item_len]))
                item_out_len = self.model_wrapper.output_sequence_length(item_len)
                torch.testing.assert_close(
                        output[i, :item_out_len, :],
                        output_wo_padding[0, :, :])
