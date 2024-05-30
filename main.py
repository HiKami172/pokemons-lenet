"""
Pokemon CNN classification model train/test
"""

from typing import List, Tuple, Dict
from PIL import Image
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


class PokemonDataset(Dataset):
    """
    Dataset for Pokemon CNN classification model
    """

    def __init__(self, root_dir):
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class LeNet(nn.Module):
    """
    LeNet Model for Pokemon classification.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 150)

    def forward(self, x):
        """
        Torch model layer forward function.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_datasets() -> Tuple[PokemonDataset, PokemonDataset]:
    """
    Load train/test datasets from pokemons dir.
    :return: train/test datasets.
    """
    train_dataset = PokemonDataset(root_dir='pokemons/train')
    test_dataset = PokemonDataset(root_dir='pokemons/test')
    return train_dataset, test_dataset


def get_class_labels() -> List[str]:
    """
    Get list of class labels.
    :return: labels
    """
    train_dataset = PokemonDataset(root_dir='pokemons/train')
    return train_dataset.dataset.classes


def train_model(train_dataset: PokemonDataset,
                test_dataset: PokemonDataset,
                num_epochs: int = 10,
                batch_size: int = 32) -> Dict:
    """
    Train LeNet model with CE loss function and Adam optimizer.
    :return: Dict with trained model and history.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = correct / total

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return {
        'model': model,
        'history': history
    }


def load_model_and_predict(model_path: str, paths: List[str]) -> List[str]:
    """
    Load trained model and predict on test data.
    :param model_path: saved model path
    :param paths: paths to images to test on
    :return: labels
    """
    class_names = get_class_labels()
    model = LeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    images = [transform(Image.open(image_path).convert('RGB')) for image_path in paths]
    images = torch.stack(images)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    predicted_labels = [class_names[p] for p in predicted]
    return predicted_labels


def load_dataset_and_train() -> None:
    """
    Load local dataset and train model.
    """
    train_dataset, test_dataset = load_datasets()

    result = train_model(train_dataset, test_dataset, num_epochs=30)

    model = result.get('model')
    history = result.get('history')

    torch.save(model.state_dict(), 'lenet_model.pth')
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)


def test_model() -> None:
    """
    Test model with predefined images.
    """
    image_paths = [
        "pokemons/test/Abra/34532bb006714727ade4075f0a72b92d.jpg",
        "pokemons/test/Cubone/9d5269c2392b49b6b8f37e15ff284351.jpg",
        "pokemons/test/Pikachu/00000009.png",
        "pokemons/test/Venonat/11a1fcd8f49a436fb14361b20ab2f571.jpg",
        "pokemons/test/Wigglytuff/9ba2b3c0b13243dc9c909e7437849c3f.jpg"
    ]
    labels = load_model_and_predict("lenet_model.pth", image_paths)
    print('\n'.join(f"{image}: {label}" for image, label in zip(image_paths, labels)))


if __name__ == '__main__':
    load_dataset_and_train()
    test_model()
