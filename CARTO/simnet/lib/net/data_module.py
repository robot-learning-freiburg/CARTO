import pytorch_lightning as pl

from CARTO.simnet.lib.net import common


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams, train_dataset=None, preprocess_func=None):
        super().__init__()

        # Using the same hyperparmeter saving method as the model module
        # doesn't work, so just assign to some other variable for now.
        self.params = hparams
        self.train_dataset = train_dataset
        self.preprocess_func = preprocess_func

    def train_dataloader(self):
        return common.get_loader(
            self.params,
            "train",
            preprocess_func=self.preprocess_func,
            datapoint_dataset=self.train_dataset,
        )

    def val_dataloader(self):
        return common.get_loader(
            self.params, "val", preprocess_func=self.preprocess_func
        )

    def test_dataloader(self):
        return common.get_loader(
            self.params, "test", preprocess_func=self.preprocess_func
        )
