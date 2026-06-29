"""
Training Pipeline for Satellite Infrared-to-Optical Image Translation.

Loads the training and validation datasets, configures training hyperparameters,
runs the optimization loop for the DualStreamIR2RGB model, and outputs loss plots
and model checkpoints.
"""

import os
import argparse
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from core.engine import DualStreamIR2RGB
from core.dataset import SatelliteDataset


def save_loss_curve_cv2(train_losses: list[float], val_losses: list[float], output_path: str = "loss_curve.png"):
    """
    Draw and save a loss curve graph as a PNG image using OpenCV.
    Avoids dependency on matplotlib.
    """
    w, h = 800, 400
    # Create dark-gray background canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = [20, 24, 30]  # #14181E dark theme color
    
    # Check if we have data to plot
    if not train_losses:
        cv2.imwrite(output_path, canvas)
        return
        
    num_epochs = len(train_losses)
    all_losses = train_losses + val_losses
    max_loss = max(all_losses) if all_losses else 1.0
    min_loss = min(all_losses) if all_losses else 0.0
    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0
    
    # Padding around the graph
    pad_left = 80
    pad_right = 30
    pad_top = 40
    pad_bottom = 50
    
    graph_w = w - pad_left - pad_right
    graph_h = h - pad_top - pad_bottom
    
    # Draw Grid Lines
    for i in range(5):
        y_grid = int(pad_top + i * (graph_h / 4))
        cv2.line(canvas, (pad_left, y_grid), (w - pad_right, y_grid), (48, 54, 61), 1)
        val = max_loss - i * (loss_range / 4)
        cv2.putText(canvas, f"{val:.4f}", (10, y_grid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (88, 166, 255), 1)
        
    # Draw X-axis Epoch labels
    epoch_step = max(1, num_epochs // 5)
    for i in range(0, num_epochs, epoch_step):
        x_grid = int(pad_left + i * (graph_w / max(1, num_epochs - 1)))
        cv2.line(canvas, (x_grid, pad_top), (x_grid, h - pad_bottom), (48, 54, 61), 1)
        cv2.putText(canvas, f"Ep {i+1}", (x_grid - 15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 148, 158), 1)
        
    # Draw axis lines
    cv2.line(canvas, (pad_left, pad_top), (pad_left, h - pad_bottom), (139, 148, 158), 2)
    cv2.line(canvas, (pad_left, h - pad_bottom), (w - pad_right, h - pad_bottom), (139, 148, 158), 2)
    
    # Plotting helper
    def get_coords(epoch_idx, loss_val):
        x = int(pad_left + epoch_idx * (graph_w / max(1, num_epochs - 1)))
        y = int(h - pad_bottom - ((loss_val - min_loss) / loss_range) * graph_h)
        return x, y
        
    # Draw Train Loss curve (Green line)
    train_pts = [get_coords(i, l) for i, l in enumerate(train_losses)]
    for i in range(len(train_pts) - 1):
        cv2.line(canvas, train_pts[i], train_pts[i+1], (46, 160, 67), 2, cv2.LINE_AA)
        
    # Draw Val Loss curve (Cyan line)
    val_pts = [get_coords(i, l) for i, l in enumerate(val_losses)]
    for i in range(len(val_pts) - 1):
        cv2.line(canvas, val_pts[i], val_pts[i+1], (88, 166, 255), 2, cv2.LINE_AA)
        
    # Draw Legend
    cv2.rectangle(canvas, (w - 200, 10), (w - 10, 45), (30, 35, 41), -1)
    cv2.rectangle(canvas, (w - 200, 10), (w - 10, 45), (48, 54, 61), 1)
    # Train legend
    cv2.line(canvas, (w - 190, 20), (w - 165, 20), (46, 160, 67), 2)
    cv2.putText(canvas, "Train Loss (MSE)", (w - 155, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 237, 243), 1)
    # Val legend
    cv2.line(canvas, (w - 190, 35), (w - 165, 35), (88, 166, 255), 2)
    cv2.putText(canvas, "Val Loss (MSE)", (w - 155, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 237, 243), 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"Loss curve plot saved to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Train Satellite IR-to-RGB Translation model.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default="weights/ir2rgb_trained.pth", help="Path to save model checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint if available")
    parser.add_argument("--limit-steps", type=int, default=None, help="Limit number of batches processed per epoch for verification")
    args = parser.parse_args()

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Dataset loading
    train_ir_dir = os.path.join(args.data_dir, "train", "ir")
    train_rgb_dir = os.path.join(args.data_dir, "train", "rgb")
    val_ir_dir = os.path.join(args.data_dir, "val", "ir")
    val_rgb_dir = os.path.join(args.data_dir, "val", "rgb")

    try:
        train_dataset = SatelliteDataset(train_ir_dir, train_rgb_dir)
        val_dataset = SatelliteDataset(val_ir_dir, val_rgb_dir)
    except FileNotFoundError as e:
        print(f"Dataset loading failed: {e}")
        print("Please run 'python utils/dataset_setup.py' first to initialize the folders and data.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Datasets successfully loaded.")
    print(f" - Training samples: {len(train_dataset)} (Batches: {len(train_loader)})")
    print(f" - Validation samples: {len(val_dataset)} (Batches: {len(val_loader)})")

    # Instantiate Model
    model = DualStreamIR2RGB().to(device)
    
    # Resume checkpoint logic
    if args.resume and os.path.exists("weights/ir2rgb_dummy.pth"):
        print("Loading weights from weights/ir2rgb_dummy.pth as baseline...")
        try:
            model.load_state_dict(torch.load("weights/ir2rgb_dummy.pth", map_location=device))
        except Exception as e:
            print(f"Warning: Could not resume from baseline: {e}. Starting fresh.")
    elif args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming weights from existing checkpoint {args.checkpoint}...")
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting fresh.")

    # Optimizer and Loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train_losses = []
    val_losses = []

    print("\n--- Starting Training Loop ---")
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_steps = 0
        for batch_idx, (ir_imgs, rgb_imgs) in enumerate(train_loader):
            if args.limit_steps is not None and batch_idx >= args.limit_steps:
                break
            ir_imgs = ir_imgs.to(device)
            rgb_imgs = rgb_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(ir_imgs)
            loss = criterion(outputs, rgb_imgs)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_steps += 1
            
        epoch_train_loss = running_train_loss / train_steps if train_steps > 0 else 0.0
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_idx, (ir_imgs, rgb_imgs) in enumerate(val_loader):
                if args.limit_steps is not None and batch_idx >= args.limit_steps:
                    break
                ir_imgs = ir_imgs.to(device)
                rgb_imgs = rgb_imgs.to(device)
                
                outputs = model(ir_imgs)
                val_loss = criterion(outputs, rgb_imgs)
                running_val_loss += val_loss.item()
                val_steps += 1
                
        epoch_val_loss = running_val_loss / val_steps if val_steps > 0 else 0.0
        val_losses.append(epoch_val_loss)
        
        epoch_duration = time.perf_counter() - epoch_start
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
              f"| Train Loss: {epoch_train_loss:.6f} "
              f"| Val Loss: {epoch_val_loss:.6f} "
              f"| Time: {epoch_duration:.2f}s")

    # Create weights directory if missing
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    
    # Save trained checkpoint
    torch.save(model.state_dict(), args.checkpoint)
    print(f"\nTrained model weights successfully saved to: {os.path.abspath(args.checkpoint)}")

    # Plot and save loss curves
    save_loss_curve_cv2(train_losses, val_losses)
    print("Training process finished.")


if __name__ == "__main__":
    main()
