# user_ml.py – completely unmodified PyTorch training script
import torch
import torch.nn as nn
import torch.optim as optim

# Simple model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(10):
    for i in range(100):   # 100 iterations per epoch
        # Simulated forward/backward
        inputs = torch.randn(10)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"Epoch {epoch}, iter {i}")