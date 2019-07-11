import numpy as np

class SaveFeatureVectors():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, inputs, outputs):
        out = outputs.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()
