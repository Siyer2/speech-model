"""Main training script."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .model import SpeechClassifier
from .wandb_utils import WandBLogger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dummy_data(batch_size: int, num_batches: int = 10):
    """Create dummy data for demonstration.

    Args:
        batch_size: Batch size
        num_batches: Number of batches to generate

    Returns:
        DataLoader with dummy audio features and labels
    """
    # Simulate mel spectrogram features: (batch, seq_len=100, features=80)
    seq_len = 100
    input_dim = 80
    num_samples = batch_size * num_batches

    x = torch.randn(num_samples, seq_len, input_dim)
    # Random labels for 5 classes
    y = torch.randint(0, 5, (num_samples,))

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for _batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    """Main training function."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "trains.yaml"
    config = Config.from_yaml(config_path)

    print(f"Loaded configuration from {config_path}")
    print(f"Training for {config.training.epochs} epochs")

    # Set seed
    set_seed(config.training.seed)

    # Initialize WandB
    wandb_logger = WandBLogger(config, config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SpeechClassifier(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Create dummy data
    train_loader = create_dummy_data(
        batch_size=config.training.batch_size,
        num_batches=20,
    )

    print("\nStarting training...")
    print("-" * 50)

    # Training loop
    for epoch in range(config.training.epochs):
        loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device)

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train/loss": loss,
            "train/accuracy": accuracy,
        }
        wandb_logger.log(metrics, step=epoch)

        print(
            f"Epoch {epoch + 1}/{config.training.epochs} - "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    print("-" * 50)
    print("Training complete!")

    # Cleanup
    wandb_logger.finish()


if __name__ == "__main__":
    main()
