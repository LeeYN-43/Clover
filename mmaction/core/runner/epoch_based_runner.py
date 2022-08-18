import time
import warnings
import platform
import shutil
import os.path as osp
import torch
from torch.optim import Optimizer

import mmcv
from mmcv.runner import RUNNERS, EpochBasedRunner, get_dist_info, get_host_info
from mmcv.runner.checkpoint import is_module_wrapper, weights_to_cpu, get_state_dict


try:
    from prettytable import PrettyTable
except:
    import subprocess
    status, output = subprocess.getstatusoutput('pip3 install --user prettytable')
    if status == 0:
        from prettytable import PrettyTable
    else:
        warnings.warn(f'Failed to install prettytable! More information: {output}')


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """

    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    mmcv.mkdir_or_exist(osp.dirname(filename))
    # immediately flush buffer
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)
        f.flush()


@RUNNERS.register_module()
class TimerEpochBasedRunner(EpochBasedRunner):
    """
    Add each epoch time based on EpochBasedRunner, and use prettyTable to display model-related information
    """

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
        assert len(data_loaders) == len(workflow)
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
                    epoch_runner(data_loaders[i], **kwargs)
                    te = time.time()
                    self.logger.info(f'{mode} epoch {self.epoch} finished in {te - ts} seconds')

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def parameter_info(self):
        rank, _ = get_dist_info()
        if rank == 0:
            if is_module_wrapper(self.model):
                model = self.model.module
            else:
                model = self.model
            model.train()

            # param_groups = self.optimizer.state_dict()['param_groups']  # comment out  
            param_groups = self.optimizer.param_groups  # add  , for torch1.10
            id2lrdecay = {id(p): pg for pg in param_groups for p in pg['params']}

            opt_def = self.optimizer.defaults
            lr_def = opt_def.get('lr', 0.1)
            wd_def = opt_def.get('weight_decay', 0.1)
            self.logger.info(f'optimizer.defaults: {opt_def}')

            table = PrettyTable(
                ['Layer Name', 'Weight Shape', 'Data Type', 'requires_grad', 'lr_mult', 'decay_mult'])

            for k, v in model.named_parameters():
                lr_mult, decay_mult = 1, 1
                pg = id2lrdecay.get(id(v), None)
                if pg is not None:
                    if 'lr_mult' in pg and 'decay_mult' in pg:
                        lr_mult, decay_mult = pg.get('lr_mult'), pg.get('decay_mult')
                    elif 'lr' in pg and 'weight_decay' in pg:
                        lr_mult = pg.get('lr', 0.1) / lr_def
                        decay_mult = pg.get('weight_decay', 0.1) / wd_def

                table.add_row([k, tuple(v.shape), v.dtype, v.requires_grad, lr_mult, decay_mult])
            table.align = 'l'
            self.logger.info('\n' + table.get_string())
            del model

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=False):
        """
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

