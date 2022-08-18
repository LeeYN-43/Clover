import torch.nn as nn
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd, _InstanceNorm


@OPTIMIZER_BUILDERS.register_module()
class CTNOptimizerConstructor(DefaultOptimizerConstructor):
    """
    Optimizer constructor in CT-Net.
    """

    def add_params(self, params, model):
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        fc_lr5 = self.paramwise_cfg.get('fc_lr5', False)
        extra_id = self.paramwise_cfg.get('extra_id', None)
        partial_bn = self.paramwise_cfg.get('partial_bn', True)
        modality = 'RGB'
        special_ids = list()

        first_conv_weight = list()
        first_conv_bias = list()
        normal_weight = list()
        normal_bias = list()
        lr5_weight = list()
        lr10_bias = list()
        bn = list()
        custom_ops = list()

        conv_cnt = 0
        bn_cnt = 0

        extra_weight = list()
        extra_bias = list()
        extra_bn = list()

        for n, m in model.named_modules():
            is_custom = False
            if isinstance(m, (_ConvNd, _BatchNorm, _InstanceNorm, nn.GroupNorm, nn.Linear)):
                for key in sorted_keys:
                    if key in n:
                        is_custom = True
                        for name, param in m.named_parameters(recurse=False):
                            param_group = {'params': [param]}
                            lr_mult = custom_keys[key].get('lr_mult', 1)
                            decay_mult = custom_keys[key].get('decay_mult', 1)
                            param_group['lr_mult'] = lr_mult
                            param_group['decay_mult'] = decay_mult
                            param_group['lr'] = self.base_lr * lr_mult
                            if self.base_wd is not None:
                                param_group['weight_decay'] = self.base_wd * decay_mult
                            params.append(param_group)
                        break

            if not is_custom:
                if isinstance(m, _ConvNd):
                    m_params = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(m_params[0])
                        if len(m_params) == 2:
                            first_conv_bias.append(m_params[1])
                    else:
                        if id(m_params[0]) not in special_ids:
                            # not add special convolution whose gradient is not updated
                            if extra_id and id(m_params[0]) in extra_id:
                                extra_weight.append(m_params[0])
                                if len(m_params) == 2:
                                    extra_bias.append(m_params[1])
                            else:
                                normal_weight.append(m_params[0])
                                if len(m_params) == 2:
                                    normal_bias.append(m_params[1])
                elif isinstance(m, nn.Linear):
                    m_params = list(m.parameters())
                    if fc_lr5:
                        lr5_weight.append(m_params[0])
                    else:
                        normal_weight.append(m_params[0])
                    if len(m_params) == 2:
                        if fc_lr5:
                            lr10_bias.append(m_params[1])
                        else:
                            normal_bias.append(m_params[1])
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not partial_bn or bn_cnt == 1:
                        m_params = list(m.parameters())
                        if extra_id and id(m_params[0]) in extra_id:
                            extra_bn.extend(m_params)
                        else:
                            bn.extend(m_params)
                elif isinstance(m, nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not partial_bn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        params.extend([
            {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
            # for extra, set lr_mult for different lr
            {'params': extra_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "extra_weight"},
            {'params': extra_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "extra_bias"},
            {'params': extra_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "extra BN scale/shift"},
            # {'params': extra_weight, 'lr_mult': 3, 'decay_mult': 1,
            # 'name': "extra_weight"},
            # {'params': extra_bias, 'lr_mult': 6, 'decay_mult': 0,
            # 'name': "extra_bias"},
            # {'params': extra_bn, 'lr_mult': 3, 'decay_mult': 0,
            # 'name': "extra BN scale/shift"},
            # {'params': extra_weight, 'lr_mult': 5, 'decay_mult': 1,
            # 'name': "extra_weight"},
            # {'params': extra_bias, 'lr_mult': 10, 'decay_mult': 0,
            # 'name': "extra_bias"},
            # {'params': extra_bn, 'lr_mult': 5, 'decay_mult': 0,
            # 'name': "extra BN scale/shift"},
        ])
