import torch
import numpy as np

def validate_model(model, val_loader, criterion, verbose=True):
    device = next(model.parameters()).device  # Get model's device
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for embeddings, labels in val_loader:
            # Move data to the same device as the model
            embeddings, labels = embeddings.to(device), labels.to(device).view(-1,1)
            
            # Forward pass
            outputs = model(embeddings)  # Squeeze to match labels
            loss = criterion(outputs, labels.float())
            
            # Metrics
            val_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = total_correct / total_samples
    if verbose:
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    return avg_val_loss, val_accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3, min_delta=0.001, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device).view(-1,1)
            
            optimizer.zero_grad()
            outputs = model(embeddings)  # Shape: [batch_size]
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Epoch statistics
        avg_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")

        # Validation
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, verbose)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print("Early stopping triggered!")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies