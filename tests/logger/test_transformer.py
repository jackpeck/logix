import copy
import unittest

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForSequenceClassification

from logix import LogIX


class TestTransformerGradients(unittest.TestCase):
    def setUp(self):
        # Create a minimal configuration for BERT
        config = BertConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=10,
            dropout=0.0,
        )
        self.model = BertForSequenceClassification(config)
        self.func_model = BertForSequenceClassification(config)
        self.func_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.func_params = dict(self.func_model.named_parameters())
        self.func_buffers = dict(self.func_model.named_buffers())

        self.model.eval()
        self.func_model.eval()

    def _compute_reference_grads(self, batch):
        def compute_loss_func(_params, _buffers, _batch):
            _output = torch.func.functional_call(
                self.func_model,
                (_params, _buffers),
                args=(
                    _batch[0].unsqueeze(0),
                    _batch[1].unsqueeze(0),
                ),
            )
            _loss = F.cross_entropy(_output.logits, _batch[2].unsqueeze(0))
            return _loss

        func_compute_grad = torch.func.grad(compute_loss_func, has_aux=False)

        per_example_grads = []
        for idx in range(batch[0].shape[0]):
            single_batch = tuple(t[idx] for t in batch)
            per_example_grads.append(
                func_compute_grad(self.func_params, self.func_buffers, single_batch)
            )

        grads_dict = {}
        for name in per_example_grads[0]:
            grads_dict[name] = torch.stack(
                [per_example_grad[name] for per_example_grad in per_example_grads],
                dim=0,
            )

        return grads_dict

    def test_per_sample_gradient(self):
        # Instantiate LogIX
        logix = LogIX(project="test")
        logix.watch(self.model)

        # Input and target for batch size of 4
        input_ids = torch.randint(0, 32, (4, 10))  # Dummy token IDs
        attention_mask = torch.ones(4, 10)  # All tokens are 'real'
        labels = torch.tensor([1, 0, 1, 0])  # Dummy labels
        batch = (input_ids, attention_mask, labels)

        grads_dict = self._compute_reference_grads(batch)

        # Forward pass with original model
        logix.setup({"grad": ["log"]})
        with logix(data_id=input_ids):
            self.model.zero_grad()
            output = self.model(input_ids, attention_mask).logits
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        _, logix_grads_dict = logix.get_log()

        for module_name in logix_grads_dict:
            logix_grad = logix_grads_dict[module_name]["grad"]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(logix_grad, func_grad, atol=1e-6))

    def test_per_sample_gradient_mask(self):
        # Instantiate LogIX
        logix = LogIX(project="test")
        logix.watch(self.model)

        # Input and target for batch size of 4
        input_ids = torch.randint(0, 32, (4, 10))  # Dummy token IDs
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        labels = torch.tensor([1, 0, 1, 0])  # Dummy labels
        batch = (input_ids, attention_mask, labels)

        grads_dict = self._compute_reference_grads(batch)

        # Forward pass with original model
        logix.setup({"grad": ["log"]})
        with logix(data_id=input_ids, mask=attention_mask):
            self.model.zero_grad()
            output = self.model(input_ids, attention_mask).logits
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        _, logix_grads_dict = logix.get_log()

        for module_name in logix_grads_dict:
            logix_grad = logix_grads_dict[module_name]["grad"]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(logix_grad, func_grad, atol=1e-6))

    def test_per_sample_gradient_mask_with_gradient_checkpoint(self):
        # Instantiate LogIX
        logix = LogIX(project="test")
        logix.watch(self.model)

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Input and target for batch size of 4
        input_ids = torch.randint(0, 32, (4, 10))  # Dummy token IDs
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        labels = torch.tensor([1, 0, 1, 0])  # Dummy labels
        batch = (input_ids, attention_mask, labels)

        grads_dict = self._compute_reference_grads(batch)

        # Forward pass with original model
        logix.setup({"grad": ["log"]})
        with logix(data_id=input_ids, mask=attention_mask):
            self.model.zero_grad()
            output = self.model(input_ids, attention_mask).logits
            loss = F.cross_entropy(output, labels, reduction="sum")
            loss.backward()
        _, logix_grads_dict = logix.get_log()

        for module_name in logix_grads_dict:
            logix_grad = logix_grads_dict[module_name]["grad"]
            func_grad = grads_dict[module_name + ".weight"]
            self.assertTrue(torch.allclose(logix_grad, func_grad, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
