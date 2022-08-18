from itertools import zip_longest
import time
import torch
from tqdm import tqdm
from mmcv.runner.builder import RUNNERS
import warnings
import mmcv
from mmcv.runner import get_host_info
from .epoch_based_runner import TimerEpochBasedRunner


@RUNNERS.register_module()
class MyEpochBasedRunner(TimerEpochBasedRunner):
    """
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1
            if i >= (len(self.data_loader) - 1):
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(tqdm(self.data_loader)):  # modified  , replace "self.data_loader" with "self.data_loader.data_iter"
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            if i >= (len(self.data_loader) - 1):
                break

        self.call_hook('after_val_epoch')



@RUNNERS.register_module()
class MyEpochBasedMultiDatasetRunner(MyEpochBasedRunner):
    """Support multiple train dataloader

    """
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        max_len_idx, max_loader_len = 0, 0
        for idx, loader in enumerate(data_loader):
            max_loader_len = max(len(loader), max_loader_len)
            if max_loader_len == len(loader):
                max_len_idx = idx
        max_len_loader = data_loader[max_len_idx]
        self.data_loader = max_len_loader
        self._max_iters = self._max_epochs * len(max_len_loader)

        self.data_iter = [loader_i for loader_i in data_loader]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        short_loader = None
        for batch_idx, data_batchs in enumerate(zip_longest(*self.data_iter)):
            self._inner_iter = batch_idx
            for loader_idx, data_batch in enumerate(data_batchs):  
                if short_loader is None and data_batch is None:
                    short_loader = iter(data_loader[loader_idx])
                    data_batch = next(short_loader)
                elif short_loader is not None:
                    data_batch = next(short_loader)

                self.call_hook('before_train_iter')         
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')   
                                                      

            self._iter += 1  # total iter num
            
            if batch_idx >= (max([len(loader) - 1 for loader in data_loader])):
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        ### print parameter information
        self.parameter_info()
        # workflow is a list, consist of mode('train', 'val', 'test') and cur epoch
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    ts = time.time()
                    epoch_runner(data_loaders, **kwargs)
                    te = time.time()
                    self.logger.info(f'{mode} epoch {self.epoch} finished in {te - ts} seconds')

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
