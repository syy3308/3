import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.conf_loss = nn.BCEWithLogitsLoss()
        self.loc_loss = nn.SmoothL1Loss()

    def forward(self, conf_preds, loc_preds, conf_targets, loc_targets):
        conf_loss = self.conf_loss(conf_preds, conf_targets)
        loc_loss = self.loc_loss(loc_preds, loc_targets)
        return conf_loss + loc_loss

