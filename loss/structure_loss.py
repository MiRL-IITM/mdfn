import torch.nn as nn
from wavemix import DWTForward

class StructureLoss(nn.Module):
    def __init__(self, wave="db1", mode="zero", J=1, loss_type='huber'):
        super(StructureLoss, self).__init__()
        self.J = J
        self.mode = mode
        self.wave = wave
        self.dwt = DWTForward(J=self.J, mode=self.mode, wave=self.wave)
        self.approximate_cef_loss = nn.HuberLoss(delta=2.0)
        self.detail_cef_loss = nn.HuberLoss(delta=0.01)

    def forward(self, pred, target):
        Y1_pred, Yh_pred = self.dwt(pred)
        Y1_target, Yh_target = self.dwt(target)
        wavelet_loss = self.approximate_cef_loss(Y1_pred, Y1_target)
        for i in range(self.J):
            wavelet_loss += self.detail_cef_loss(Yh_pred[i], Yh_target[i])
        return wavelet_loss