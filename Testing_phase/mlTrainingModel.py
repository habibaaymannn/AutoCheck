"""
train.py
Real PyTorch training — MNIST digit classifier.
Zero awareness of AutoCheck — just plain PyTorch code.

Press Ctrl+C at any point to pause.
AutoCheck saves checkpoint. Run again to resume.

Requirements:
    pip install torch torchvision
"""
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Model ─────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ── Setup ─────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model     = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

criterion = nn.CrossEntropyLoss()

NUM_EPOCHS  = 20
global_step = 0

print("Starting MNIST training... (press Ctrl+C to pause)")
print(f"{'epoch':>6} {'batch_idx':>10} {'global_step':>12} {'loss':>10}")
print("-" * 45)

# ── Training loop ─────────────────────────────────────────────────────

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()

        global_step += 1

        # print every 100 steps
        if global_step % 100 == 0:
            print(
                f"{epoch:>6} {batch_idx:>10} {global_step:>12} "
                f"{loss.item():>10.4f}"
            )

    scheduler.step()
    print(f"  → epoch {epoch} done | lr={scheduler.get_last_lr()[0]:.6f}")

print("\nTraining complete!")