import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Параметри
DATA_DIR = 'dataset'
BATCH_SIZE = 64
NUM_EPOCHS = 10
IMAGE_SIZE = 224
MODEL_PATH = 'model.pt'
CLASS_MAP_PATH = 'class_to_idx.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформації
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Завантаження даних
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Збереження мапи класів
with open(CLASS_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(dataset.class_to_idx, f, ensure_ascii=False, indent=2)

# Модель
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Лог для графіків
train_losses = []
val_accuracies = []

# Тренування
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Train loss: {avg_loss:.4f}")

    # Валідація
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
    acc = correct / total
    val_accuracies.append(acc)
    print(f"Validation accuracy: {acc:.2%}")

# Збереження моделі
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Модель збережено в {MODEL_PATH}")

# Побудова графіків
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Матриця плутанини
cm = confusion_matrix(y_true, y_pred)
labels = list(dataset.class_to_idx.keys())
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
