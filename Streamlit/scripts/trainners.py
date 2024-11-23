import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import copy



# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataloader
def get_train_loaders(train_dir, val_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_test_loader(test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience=5, warmup_steps=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_accuracy = 0.0
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.early_stop = False
        self.counter = 0
        self.best_score = None

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for step, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # warm-up steps
                if step < self.warmup_steps:
                    lr_scale = min(1.0, float(step + 1) / self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr_scale * self.optimizer.defaults['lr']

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)
            val_loss, val_accuracy = self.validate()

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # early stopping
            self._early_stopping(val_loss)
            if self.early_stop:
                print("Early stopping triggered")
                break

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return val_loss / len(self.val_loader), accuracy

    def _early_stopping(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
