import abc
import torch
import torch.nn as nn

class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))

    def export_onnx(self, sample_input, path, opset_version=11):
        self.eval()
        torch.onnx.export(
            self,
            sample_input,
            path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"]
        )

    def export_torchscript(self, sample_input, path):
        self.eval()
        traced = torch.jit.trace(self, sample_input)
        traced.save(path)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_layers(self, names=None):
        if names is None:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(k in name for k in names):
                    param.requires_grad = False

    def unfreeze_layers(self, names=None):
        if names is None:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(k in name for k in names):
                    param.requires_grad = True

    def create_optimizer(self, opt_type="adam", lr=1e-3, weight_decay=1e-4, momentum=0.9):
        if opt_type.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if opt_type.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return None

    def create_scheduler(self, optimizer, sched_type="step", step_size=10, gamma=0.1, T_max=50):
        if sched_type.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if sched_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        return None
