import os.path as osp
import subprocess
import warnings
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, CheckpointHook
from mmaction.utils import hexists, hmkdir, hdelete



def run_cmd(_cmd, logger, text, return_status=False, mute=True):
    status, output = subprocess.getstatusoutput(_cmd)
    if status == 0 and not mute:
        logger.info(text)
    elif status:
        logger.error(output)
    if return_status:
        return status


@HOOKS.register_module()
class MYCheckpointHook(CheckpointHook):
    """
    """

    def __init__(self,
                 save_root=None,
                 best_save_root=None,
                 mute=True,  # dont print log info
                 del_non_latest=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_root = save_root
        self.best_save_root = best_save_root
        self.del_non_latest = del_non_latest
        if 'del_local_ckpt' in kwargs:
            warnings.warn(f'del_local_ckpt has been deprecated! '
                          f'Checkpoints will be automatically saved at: {save_root}')
        self.mute = mute
        self.is_first = True 

    def sync_log(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        if self.is_first and self.save_root is not None:
            dirname = osp.basename(self.out_dir)
            self.out_dir = self.save_root + dirname
            if self.best_save_root is not None:
                runner.hdfs_work_dir = self.best_save_root
                if not hexists(self.best_save_root):
                    hmkdir(self.best_save_root)
            else:
                runner.hdfs_work_dir = self.out_dir

            if not hexists(self.out_dir):
                hmkdir(self.out_dir)

            self.is_first = False

    @master_only
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""

        self.sync_log(runner)

        runner.save_checkpoint(out_dir=self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.del_non_latest:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1

            prev_ckpt = current_ckpt - self.interval
            if prev_ckpt > 0:
                filename_tmpl = self.args.get('filename_tmpl', name)
                try:
                    hdelete(osp.join(self.out_dir, filename_tmpl.format(prev_ckpt)))
                except Exception as e:
                    runner.logger.info(e)

        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = osp.join(
                self.out_dir, cur_ckpt_filename)

    def after_val_epoch(self, runner):
        if runner.rank == 0:
            try:
                self.sync_log(runner)
            except Exception as e:
                runner.logger.info(e)

    def after_run(self, runner):
        if runner.rank == 0:
            try:
                self.sync_log(runner)
            except Exception as e:
                runner.logger.info(e)
