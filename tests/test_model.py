import unittest

import torch

from model import TinyGPT, TinyGPTConfig, count_parameters


class TinyGPTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TinyGPTConfig(vocab_size=64, context_length=8, n_layers=2, d_model=24, n_heads=3, ffn_hidden=96, dropout=0.0)
        self.model = TinyGPT(self.config).eval()

    def test_logits_and_loss_shape(self) -> None:
        tokens = torch.randint(0, self.config.vocab_size, (2, self.config.context_length))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, self.config.context_length, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask_blocks_future_tokens(self) -> None:
        first = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        second = first.clone()
        second[:, 5:] = torch.tensor([10, 11, 12])
        first_logits, _ = self.model(first)
        second_logits, _ = self.model(second)
        torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5])

    def test_lm_head_and_embedding_share_weights(self) -> None:
        self.assertEqual(self.model.lm_head.weight.data_ptr(), self.model.token_embedding.weight.data_ptr())

    def test_parameter_count_is_tiny_scale(self) -> None:
        self.assertEqual(count_parameters(TinyGPT(TinyGPTConfig())), 13_808_640)

    def test_generate_rejects_non_positive_top_k(self) -> None:
        tokens = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            self.model.generate(tokens, max_new_tokens=1, top_k=0)


if __name__ == "__main__":
    unittest.main()
