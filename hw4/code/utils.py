import os
import torch


def save_checkpoint(state, is_best, fileDir: str):
    os.makedirs(fileDir, exist_ok=True)
    filepath = os.path.join(fileDir, "last.pth")
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(fileDir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, ckpt_path: str, optimizer=None, scheduler=None, map_location="cuda"):
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint.get('epoch', 0)

    return model, optimizer, scheduler, start_epoch


class Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        err = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(err)
        return loss
