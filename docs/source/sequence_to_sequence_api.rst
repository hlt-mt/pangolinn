Sequence-to-sequence Package
_____________________________

This package contains test suites for sequence-to-sequence models.
To test your model, in your `tests` directory create a new class
that extends :py:class:`pangolinn.seq2seq.PangolinnSeq2SeqModuleWrapper`
and builds an instance of model to be tested. Then, create a test suite
that inherits from the pangolinn tester you want to use and set the
attribute :py:attr:`module_wrapper_class` to the name of the wrapper
class of your module. And you are done! You can run the tests for your model.

Example Usage
=============

This example tests that the PyTorch implementation of the Transformer
decoder is autoregressive. First, create a
:py:class:`pangolinn.seq2seq.PangolinnSeq2SeqModuleWrapper` for it.

.. code-block:: python

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

Then, create a test suite that uses pangolinn
:py:class:`pangolinn.seq2seq.CausalTestCase` and inform it about
the wrapper class to be used.

.. code-block:: python

   class CausalDecoderTestCase(seq2seq.CausalTestCase):
       module_wrapper_class = TransformerDecoderWrapper

API Reference
=============

.. automodule:: pangolinn.seq2seq
     :members:
