import torch


def save_checkpoint(val_loss_epoch,ckpt,model):
    global BEST_LOSS
    if val_loss_epoch < BEST_LOSS:
      BEST_LOSS = val_loss_epoch
      torch.save(model.state_dict(),ckpt)