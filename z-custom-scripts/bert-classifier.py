"""
This script will log the train loss, train accuracy, validation loss, 
validation accuracy as scalars for each epoch and also log them to tensorboard so 
that you can visualize the progress. 
You can start the tensorboard by running tensorboard --logdir=logs/ 
(if you set the log_dir to logs in script) command from the command line and
 then go to the http://localhost:6006/ in the browser.
This code was written by ChatGPT
"""

import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Define a function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the data CSV file")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Pretrained model name or path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs")
    args = parser.parse_args()
    return args

# Load the data
args = get_args()
df = pd.read_csv(args.data)

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(args.model)

# Convert the text data to input ids and attention masks
train_input_ids = tokenizer.batch_encode_plus(train_df["text"], pad_to_max_length=True, return_attention_masks=True)["input_ids"]
train_labels = train_df["label"]
test_input_ids = tokenizer.batch_encode_plus(test_df["text"], pad_to_max_length=True, return_attention_masks=True)["input_ids"]
test_labels = test_df["label"]

# Create the data loaders
train_data = TensorDataset(torch.tensor(train_input_ids), torch.tensor(train_labels))
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_data = TensorDataset(torch.tensor(test_input_ids), torch.tensor(test_labels))
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained(args.model, num_labels=len(set(train_labels)))
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()

# Initialize the TensorBoard writer
writer = SummaryWriter(args.log_dir)

# Train the model
for epoch in range(args.epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for input_ids, labels in train_dataloader:
        input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")

        # Perform the forward pass
        outputs = model(input_ids, labels=labels)
        loss = loss_fn(outputs[0], labels)

        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the train accuracy
        train_loss += loss.item()
        train_acc += (outputs[1].argmax(1) == labels).sum().item()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    # Log the training loss and accuracy to TensorBoard
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_acc", train_acc, epoch)

    # Compute the validation accuracy
    val_loss = 0
    val_acc = 0
    model.eval()
    for input_ids, labels in test_dataloader:
        input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        labels = labels.to("cuda" if torch.cuda.is_available() else "cpu")
        # Compute the validation loss and accuracy
        val_loss += loss.item()
        val_acc += (outputs[1].argmax(1) == labels).sum().item()
    val_loss /= len(test_dataloader)
    val_acc /= len(test_dataloader)
    
    # Log the validation loss and accuracy to TensorBoard
    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_acc", val_acc, epoch)
    
# Close the TensorBoard writer
writer.close()


