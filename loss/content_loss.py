import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self, loss_type='huber', delta=1.0):
        """
        Args:
            loss_type (str): 'mse', 'l1', or 'huber'
            delta (float): delta parameter for Huber loss
        """
        super(ContentLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, input, target):
        return self.loss_fn(input, target)