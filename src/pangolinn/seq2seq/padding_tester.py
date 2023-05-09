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
import torch
from torch import LongTensor

from pangolinn.seq2seq.base_tester import BaseTester


class EncoderPaddingTestCase(BaseTester):
    """
    This class provides unit tests to enforce that the module to be tested properly handles
    padding, i.e. returns the same results regardless of the amount of padding present in
    the batched input sequences.

    To use it to test your network:

     1. create a `PangolinnSeq2SeqModuleWrapper` that wraps your module (e.g. `MyWrapper`);
     2. create test class that extends `EncoderPaddingTestCase`;
     3. in your test class, override the class attribute `module_wrapper_class` by setting it to
        the **class** of your wrapper (e.g., `module_wrapper_class = MyWrapper`);
    """
    def setUp(self) -> None:
        self._wrapper_setup(EncoderPaddingTestCase)

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
            rand_batch = self._rand_tensor(
                (2, max_batch_seq_len, self.module_wrapper.num_input_channels),
                self.module_wrapper.input_dtype)
            rand_batch[1, shorter_seq_len:, :] = 0
            batch_lens = torch.LongTensor([max_batch_seq_len, shorter_seq_len])
            expected_shape = [
                2,
                self.module_wrapper.output_sequence_length(max_batch_seq_len),
                self.module_wrapper.num_output_channels]
            output = self._forward_with_expected_shape(rand_batch, batch_lens, expected_shape)
            expected_shorter_seq_len = self.module_wrapper.output_sequence_length(shorter_seq_len)
            padding_area = output[1, expected_shorter_seq_len:, :]
            self.assertGreater(padding_area.numel(), 0)
            self.assertTrue(torch.all(padding_area == 0), f"non-zero entries in {padding_area}")

    def test_batch_size_does_not_matter(self):
        """
        Tests that for the same input we get the same output regardless of the amount of padding.
        """
        for max_batch_seq_len, shorter_seq_len in [(27, 13), (24, 16)]:
            rand_batch = self._rand_tensor(
                (5, max_batch_seq_len, self.module_wrapper.num_input_channels),
                self.module_wrapper.input_dtype)
            # multiple padded elements of same len
            for b_idx in [1, 2, 3]:
                rand_batch[b_idx, shorter_seq_len:, :] = 0
            # sequence of len 1
            rand_batch[4, 1:, :] = 0
            batch_lens = torch.LongTensor([
                max_batch_seq_len, shorter_seq_len, shorter_seq_len, shorter_seq_len, 1])
            expected_shape = [
                5,
                self.module_wrapper.output_sequence_length(max_batch_seq_len),
                self.module_wrapper.num_output_channels]
            output = self._forward_with_expected_shape(rand_batch, batch_lens, expected_shape)
            for i in range(5):
                item_len = batch_lens[i].item()
                item_valid_tokens = rand_batch[i, :item_len, :].unsqueeze(0)
                output_wo_padding = self.module_wrapper.forward(
                    item_valid_tokens, LongTensor([item_len]))
                item_out_len = self.module_wrapper.output_sequence_length(item_len)
                torch.testing.assert_close(
                        output[i, :item_out_len, :],
                        output_wo_padding[0, :, :])
