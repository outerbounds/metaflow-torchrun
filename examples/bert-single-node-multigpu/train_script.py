#!/usr/bin/env python3
"""
BERT Multi-GPU Training Example using torchrun and PyTorch DDP

To run on 4 GPUs:
torchrun --nproc_per_node=4 bert_multi_gpu.py

You can also specify other torchrun parameters:
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 bert_multi_gpu.py
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def setup_distributed(args):
    """Initialize the distributed environment."""
    # torchrun sets these environment variables
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])

    # Initialize the process group
    dist.init_process_group(backend=args.backend)
    torch.cuda.set_device(args.local_rank)
    
    print(f"Initialized process {args.rank}/{args.world_size} (local_rank: {args.local_rank})")

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

class BertClassifier(nn.Module):
    """BERT-based text classifier."""
    
    def __init__(self, num_classes, pretrained_model="bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension which tokenizer adds
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

def load_data(tokenizer, args):
    """Load and prepare the dataset."""
    # Load a sample dataset (using IMDB for this example)
    dataset = load_dataset("imdb")
    
    train_texts = dataset["train"]["text"][:args.max_samples] if args.max_samples > 0 else dataset["train"]["text"]
    train_labels = dataset["train"]["label"][:args.max_samples] if args.max_samples > 0 else dataset["train"]["label"]
    
    val_texts = dataset["test"]["text"][:args.max_samples//10] if args.max_samples > 0 else dataset["test"]["text"][:5000]
    val_labels = dataset["test"]["label"][:args.max_samples//10] if args.max_samples > 0 else dataset["test"]["label"][:5000]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    # Create distributed samplers for the datasets
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed
    )
    
    # Create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

def train_epoch(model, train_loader, optimizer, criterion, epoch, args):
    """Train for one epoch."""
    model.train()
    train_sampler = train_loader.sampler
    train_sampler.set_epoch(epoch)  # Important for proper shuffling in distributed training
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = None
    if args.rank == 0:  # Only show progress on the main process
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
    
    for batch in train_loader:
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1), "acc": 100 * correct / total})
    
    if progress_bar is not None:
        progress_bar.close()
    
    # Gather metrics from all processes
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device='cuda')
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0].item() / metrics[2].item()
    accuracy = 100 * metrics[1].item() / metrics[2].item()
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, args):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Gather metrics from all processes
    metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device='cuda')
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    avg_loss = metrics[0].item() / len(val_loader) / args.world_size
    accuracy = 100 * metrics[1].item() / metrics[2].item()
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, args):
    """Save a checkpoint (only from main process)."""
    if args.rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Note: model.module to get the actual model from DDP
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved at epoch {epoch}")

def main():
    parser = argparse.ArgumentParser(description="BERT Multi-GPU Training with torchrun")
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process.")
    parser.add_argument("--world_size", type=int, default=1, help="World size (total number of processes).")
    parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend to use.")
    
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max number of training samples (for quick testing).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints.")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize distributed environment
    setup_distributed(args)
    
    # Create output directory if it doesn't exist
    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training BERT on {args.world_size} GPUs")
    
    # Load tokenizer and create model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertClassifier(num_classes=args.num_classes, pretrained_model=args.model_name)
    
    # Move model to GPU
    model = model.cuda()
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Load datasets
    train_loader, val_loader, train_sampler = load_data(tokenizer, args)
    
    # Training loop
    for epoch in range(args.epochs):
        if args.rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, args)
        val_loss, val_acc = validate(model, val_loader, criterion, args)
        
        if args.rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch in this example
            save_checkpoint(model, optimizer, epoch + 1, args)
    
    # Clean up
    cleanup()
    
    if args.rank == 0:
        print("Training completed!")

if __name__ == "__main__":
    main()