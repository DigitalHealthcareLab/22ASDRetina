import torch.nn as nn

class GradCamModel(nn.Module) : 
    def __init__(self, model) : 
        super().__init__()
        for p in model.parameters() : 
            p.reguires_grad = True        

        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.pretrained = model
        
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))

            
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        return out, self.selected_out