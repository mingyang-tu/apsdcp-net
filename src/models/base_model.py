import torch
import torch.nn as nn
import torch.nn.functional as F

from .archs import ANet, RefineNet, DeepLab, gUNet_T

from .model_util import multi_windowed_dcp


class APSDCP(nn.Module):
    def __init__(self):
        super().__init__()

        self.dcp_windows = [3, 7, 15, 31, 63, 127, 255]
        self.num_windows = len(self.dcp_windows)
        self.omega = 0.9

        self.pssnet = DeepLab(self.num_windows)
        self.t_refinenet = gUNet_T(4, 1)
        self.anet = ANet()

    def forward(self, I):
        # normalize to -1 ~ 1
        x = I * 2 - 1

        # get A
        A = self.anet(x)
        A = A * 0.5 + 0.5

        # get weights
        out = self.pssnet(x)
        weights = F.softmax(out, dim=1)

        # get DCPs
        dcp = torch.sum(multi_windowed_dcp(I / A, self.dcp_windows) * weights, dim=1, keepdim=True)

        # get t
        t = 1 - self.omega * dcp

        # refine t
        t_refine = t * 2 - 1
        t_refine = self.t_refinenet(torch.cat([x, t_refine], dim=1))

        t = t.clamp(0.1, 1)
        t_refine = t_refine.clamp(0.1, 1)

        # recover J
        J = (I - A) / t + A
        J_refine = (I - A) / t_refine + A

        return J, J_refine, t, t_refine, weights

    def fix_pssnet(self):
        for param in self.pssnet.parameters():
            param.requires_grad = False


class APSDCP_Refine(nn.Module):
    def __init__(self, pretrained_apsdcp_path=None):
        super().__init__()

        self.apsdcp = APSDCP()
        self.refinenet = RefineNet()

        if pretrained_apsdcp_path is not None:
            print(f"Load pretrained APSDCP from {pretrained_apsdcp_path}")
            self.apsdcp.load_state_dict(torch.load(pretrained_apsdcp_path, weights_only=True, map_location=torch.device("cpu")))
            for param in self.apsdcp.parameters():
                param.requires_grad = False

    def forward(self, I):
        J, J_refine, t, t_refine, weights = self.apsdcp(I)

        j_norm = J_refine * 2 - 1
        i_norm = I * 2 - 1

        out = self.refinenet(j_norm, i_norm)
        out = out * 0.5 + 0.5
        return out
