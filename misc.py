
import torch


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# this function was copied from torchtext source since I didn't want to install torchtext
def to_map_style_dataset(iter_data):
    r"""Convert iterable-style dataset to map-style dataset.
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):
        def __init__(self, iter_data) -> None:
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)