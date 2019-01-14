import torch


class WarmupAndDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.d_model**-0.5 * min(self.last_epoch**-0.5, self.last_epoch * self.warmup_steps**-1.5)
            for base_lr in self.base_lrs]
