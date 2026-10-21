"""CPU configuration and UTF-8 batch-stream tests."""
import sys
import types
import unittest
from pathlib import Path

import torch
from data import build_byte_batch_iterator
from continuum_attention import ContinuumDecoderConfig, ContinuumDecoderLM
from runtime import count_parameters


class FakeDataset:
    def __init__(self, items):
        self.items = items

    def shuffle(self, **kwargs):
        return self

    def __iter__(self):
        return iter(self.items)


class TestPortableRuntime(unittest.TestCase):
    def test_parameter_budget(self):
        model = ContinuumDecoderLM(ContinuumDecoderConfig())
        self.assertEqual(count_parameters(model), {"total": 101205120, "trainable": 101205120})

    def test_utf8_packing(self):
        fake = types.ModuleType('datasets')
        fake.load_dataset = lambda **kwargs: FakeDataset([{'text':'ab'}, {'text':'éxyz'}, {'text':'12345'}])
        old = sys.modules.get('datasets')
        sys.modules['datasets'] = fake
        try:
            it = build_byte_batch_iterator(dataset_name='text', data_files='unused.txt',
                  streaming=True, shuffle_buffer=0, block_size=4, batch_size=1,
                  infinite=False, device='cpu')
            batches = list(it)
            bytestream = ('ab\néxyz\n12345\n').encode('utf-8')
            expected = [list(bytestream[i:i+4]) for i in range(0,len(bytestream)-3,4)]
            self.assertEqual([batch.tolist()[0] for batch in batches], expected)
            self.assertTrue(all(batch.dtype == torch.long for batch in batches))
        finally:
            if old is None: del sys.modules['datasets']
            else: sys.modules['datasets'] = old


if __name__ == '__main__':
    unittest.main()
