import torch
from tqdm.notebook import tqdm
from IPython.display import clear_output



@torch.compile
def forward_and_loss(model, batch, criterion):
    model.train()
    #batch is a tensor of shape [batch, seq]
    src, tgt = batch[:, :-1], batch[:, 1:]
    logits = model(src)
    return criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


def train_step(model, batch, criterion, optimizer, scaler, scheduler, accum_steps, step):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = forward_and_loss(model, batch, criterion)

    scaler.scale(loss/accum_steps).backward()

    if (step+1)%accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return loss


def save_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler


def group_decay_parameters(model, weight_decay=0.01, no_decay=['bias', 'LayerNorm.weight']):
    """
    Groups parameters for optimizer with weight decay and no weight decay.
    """
    param_optimizer = list(model.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},  # Apply weight decay to these parameters
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}  # No weight decay for these parameters
    ]
    
    return optimizer_grouped_parameters