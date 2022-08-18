from mmcv.runner import HOOKS, LrUpdaterHook
from mmcv.runner.hooks.lr_updater import annealing_linear

@HOOKS.register_module()
class LinearAnnealingLrUpdaterHook(LrUpdaterHook):
    """Linear annealing LR Scheduler decays the learning rate of each parameter
    group linearly.
    Args:
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(LinearAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_linear(base_lr, target_lr, progress / max_progress)
