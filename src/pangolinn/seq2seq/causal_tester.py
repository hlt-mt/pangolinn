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

from pangolinn.seq2seq.base_tester import BaseTester


class CausalTestCase(BaseTester):
    """
    This class provides unit tests to enforce that the module to be tested is safe to
    be used in causal models, i.e. that do not look at future elements. This property
    should be enforced for e.g. Transformer autoregressive decoders.

    To use it to test your network:

     1. create a `PangolinnSeq2SeqModuleWrapper` that wraps your module (e.g. `MyWrapper`);
     2. create test class that extends `CausalTestCase`;
     3. in your test class, override the class attribute `module_wrapper_class` by setting it to
        the **class** of your wrapper (e.g., `module_wrapper_class = MyWrapper`);
    """
    def setUp(self) -> None:
        self._wrapper_setup(CausalTestCase)

    def test_gradient_not_flowing_from_future(self):
        """
        Checks that the gradient is not backpropagated to future input time steps, which should not
        be used to compute the output.
        """
        x = self._rand_tensor(
            (1, 10, self.module_wrapper.num_input_channels), self.module_wrapper.input_dtype)
        x.requires_grad = True
        expected_shape = [
            1,
            self.module_wrapper.output_sequence_length(10),
            self.module_wrapper.num_output_channels]
        y = self._forward_with_expected_shape(x, torch.LongTensor([10]), expected_shape).abs()
        for i in range(1, 9):
            # how many of the output elements depend on a prefix made of i elements
            y_prefix_end = self.module_wrapper.output_sequence_length(i)
            grad = torch.autograd.grad(y[:, y_prefix_end, :].sum(), x, retain_graph=True)[0]
            # Checks that the gradient for the prefix of x made of i elements is not zero,
            # while it is zero for the elements after the prefix of i elements, as the gradient
            # should not propagate to the future
            self.assertGreater(grad[0, :i, :].sum().abs(), 0.0)
            self.assertAlmostEqual(grad[0, i+1:, :].sum().item(), 0.0)

    def test_not_looking_at_the_future(self):
        """
        Tests that the module masks future elements and it does not look at them.
        """
        test_len = 20
        x = self._rand_tensor(
            (5, test_len, self.module_wrapper.num_input_channels), self.module_wrapper.input_dtype)
        batch_lens = torch.LongTensor([test_len] * 5)
        expected_shape = [
            5,
            self.module_wrapper.output_sequence_length(test_len),
            self.module_wrapper.num_output_channels]
        output = self._forward_with_expected_shape(x, batch_lens, expected_shape)
        for j in range(1, 19):
            # Checks that for each of the 20 elements we obtain the same prefix in the
            # results when feeding the model with the full input sequences and the input
            # prefix truncated at that element.
            partial_lens = torch.LongTensor([j] * 5)
            partial_output = self.module_wrapper.forward(x[:, :j, :], partial_lens)
            partial_output_len = partial_output.shape[1]
            torch.testing.assert_close(
                partial_output,
                output[:, :partial_output_len, :])
