import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop

########## For individual gradient clip ###########
def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, dp=True, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.dp = dp

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)


            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        if self.dp:
                            param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(1. / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

#
# ############## For batch gradient clip ##################
# def make_optimizer_class(cls):
#     class DPOptimizerClass(cls):
#         def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
#             super(DPOptimizerClass, self).__init__(*args, **kwargs)
#
#             self.l2_norm_clip = l2_norm_clip
#             self.noise_multiplier = noise_multiplier
#             self.microbatch_size = microbatch_size
#             self.minibatch_size = minibatch_size
#
#             for group in self.param_groups:
#                 group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
#
#         def zero_microbatch_grad(self):
#             super(DPOptimizerClass, self).zero_grad()
#
#         def microbatch_step(self):
#             total_norm = 0.
#             for group in self.param_groups:
#                 for param, accum_grad in zip(group['params'], group['accum_grads']):
#                     if param.requires_grad:
#                         param.grad.data.mul_(1. / self.minibatch_size)
#                         total_norm += param.grad.data.norm(2).item() ** 2.
#                         accum_grad.add_(param.grad.data)
#             total_norm = total_norm ** .5
#             # print('Grad norm for a micro batch / B:', total_norm)
#
#             # newly added, want to track and print the individual grad norm
#             return round(total_norm * self.minibatch_size, 3)
#
#         def zero_grad(self):
#             for group in self.param_groups:
#                 for accum_grad in group['accum_grads']:
#                     if accum_grad is not None:
#                         accum_grad.zero_()
#
#         def step(self, *args, **kwargs):
#             total_norm = 0.
#             for group in self.param_groups:
#                 for param, accum_grad in zip(group['params'], group['accum_grads']):
#                     if param.requires_grad:
#                         total_norm += accum_grad.norm(2).item() ** 2.
#             total_norm = total_norm ** .5
#             # print('Grad norm of a mini batch before clipping:', total_norm)
#             clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
#             # print("Clip coefficient:", clip_coef)
#
#             # total_norm = 0.
#             for group in self.param_groups:
#                 for param, accum_grad in zip(group['params'], group['accum_grads']):
#                     if param.requires_grad:
#                         accum_grad.mul_(clip_coef)
#                         param.grad.data = accum_grad.clone()
#                         # total_norm += param.grad.data.norm(2).item() ** 2.
#                         param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
#             # total_norm = total_norm ** .5
#             # print('Grad norm of a mini batch after clipping:', total_norm)
#
#             super(DPOptimizerClass, self).step(*args, **kwargs)
#
#             # newly added, want to track and print batch gradient norm
#             return round(total_norm, 3)
#
#     return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

