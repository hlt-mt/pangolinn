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
from torch import nn, LongTensor, Tensor


class PangolinnSeq2SeqModuleWrapper:
    """
    Wrapper of your module used in pangolinn tests. To test your network, extend
    this class by implementing at least:

     - build_module;
     - num_input_channels;
     - forward.

    Depending on the behavior of your module you might need to override other methods.
    Please refer to the documentation of each method to check if you need (or not) to override it.
    """
    def __init__(self):
        self._module = self.build_module()
        self._module.eval()

    def build_module(self) -> nn.Module:
        """
        This method is responsible for building the module that you want to test.
        Override this method so that it creates an instance of the module to be tested.
        The module does not need to be initialized with specific weights.

        :return: the network you want to test.
        """
        raise NotImplementedError(
            "Please implement build_module to return the module to test")

    @property
    def num_input_channels(self) -> int:
        """
        This property has to be overridden and returns the number of channels expected in
        input by the module to be tested. It is typically the third dimension of
        sequence-to-sequence modules, in addition to the batch size and the sequence length.
        In the case of modules that expect a single integer (e.g., the text token id in text
        processing), this should be set to 1 and when overriding the forward method you can
        squeeze the last dimension before feeding the module with it.

        :return: the number of channels expected in the input tensor by the module to be tested.
        """
        raise NotImplementedError(
            "Please implement num_input_channels to return the number of channels expected by "
            "the module to be tested")

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        """
        Processes `x` with the wrapped module and returns the output.

        :param x: the tensor to be fed to the wrapped module with shape (batch, seq_len, channels)
        :param lengths: tensor of shape (batch, ) that contains the length of the valid tokens
                        for each of the sequences in the batch.
        :return: the tensor produced by the module with shape (batch, seq_len, channels)
        """
        raise NotImplementedError(
            "Please implement forward to return the output of the wrapped module to be tested")

    @property
    def sequence_downsampling_factor(self) -> int:
        """
        If the module to be tested reduces the sequence length (e.g., as strided convolutions do),
        this property states the downsampling factor.

        :return: the downsampling factor over the sequence length (e.g., in speech processing
                 Transformer-based architectures, this is often 4). By default, this is set to 1,
                 which means no downsampling.
        """
        return 1

    @property
    def num_output_channels(self) -> int:
        """
        By default, this property returns as output channels the same as the input channels.
        If your module changes the number of channels, override this method accordingly.

        :return: the number of channels produced in output by the module to be tested.
        """
        return self.num_input_channels

    def output_sequence_length(self, input_sequence_len: int) -> int:
        """
        By default, this property returns as output sequence length the `input_sequence_len`
        divided by the downsampling factor. If your module has a more complicated function
        to determine the output sequence length from the input sequence length, override
        this method accordingly.

        :param input_sequence_len: the length of a sequence to be fed to the module.
        :return: the length of the valid tokens in the output sequence.
        """
        return ((input_sequence_len - 1) // self.sequence_downsampling_factor) + 1

    @property
    def input_dtype(self) -> torch.dtype:
        """
        :return: the dtype of the input tensor expected by the module. Defaults to torch.float.
        """
        return torch.float

    @property
    def max_value_allowed(self) -> int:
        """
        :return: if `input_dtype` is set to torch.int or another integer, this determines
                 the maximum value supported as input of the module to be tested. Defaults to 10.
        """
        return 10
