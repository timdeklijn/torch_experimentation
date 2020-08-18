import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

from vgg19 import VggTim


def create_data_loader(train_folder: str, batch_size: int):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.485, 0.485],
            std=[0.229, 0.224, 0.225]
        )
    ])

    pokemon_dataset = datasets.ImageFolder(
        root=train_folder,
        transform=data_transform
    )
    dataset_loader = torch.utils.data.DataLoader(
        pokemon_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return dataset_loader, pokemon_dataset.class_to_idx


if __name__ == "__main__":
    # Run parameters
    batch_size = 4
    n_epochs = 1

    # Create dataset loader
    pokemon_dataset_loader, class_dict = create_data_loader("train/", batch_size)
    num_classes = len(class_dict)

    # Load model, loss function and optimizer
    model = VggTim(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(pokemon_dataset_loader, 0):
            # Unpack images and labels
            inputs, labels = data

            # Zero out the gradients
            optimizer.zero_grad()
            # Predictions for inputs
            outputs = model(inputs)
            # Calculate loss and update weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print some statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1:d}, {i + 1:5d}] - loss: {loss:.3f}")
                running_loss = 0.0

    print("Saing model")
    torch.save(model.state_dict(), "nets/pokemon_1.pth")
