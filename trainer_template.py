class Trainer:
    def __init__(self, pipeline: CustomPipeline, dataloader, optimizer: Optional[optim.Optimizer] = None):
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.optimizer = optimizer

    def train_step(self, batch):
        self.pipeline.model.train()
        # Perform forward pass
        outputs = self.pipeline.model(batch['input'])
        # Compute loss
        loss = self.compute_loss(outputs, batch['target'])
        # Backward pass
        loss.backward()
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def compute_loss(self, outputs, targets):
        # Implement loss computation
        return nn.MSELoss()(outputs, targets)  # Example loss function

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                self.train_step(batch)
            print(f"Epoch {epoch+1}/{num_epochs} completed.")
