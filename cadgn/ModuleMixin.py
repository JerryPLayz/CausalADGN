class ModuleMixin:
    @property
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

