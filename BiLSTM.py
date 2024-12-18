import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score
import time

seed = 27
torch.manual_seed(seed)

# Load and Resample Data
train_data = os.path.join(os.path.dirname(__file__), "hw2_train.csv")
df = pd.read_csv(train_data)
counts = df["IOB Slot tags"].str.replace("-", "_").str.split().value_counts()  # highest count is 60
least_common = counts[counts < 20] 
most_common = counts[counts >= 20]
samples = counts[counts >= 20].sum() * 3  # make highest count (20) = 60
minority = df[df["IOB Slot tags"].str.replace("-", "_").str.split().isin(least_common.index)]
majority = df[df["IOB Slot tags"].str.replace("-", "_").str.split().isin(most_common.index)]
minority_resampled = resample(minority, replace=True, n_samples=samples, random_state=27)
balanced = pd.concat([minority_resampled, majority], ignore_index=True)

x = balanced["utterances"]
y = balanced["IOB Slot tags"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=27)

# Create vocabulary and tag dictionaries
class IOBDataset(Dataset):
    def __init__(self, x, y=None, token_vocab=None, tag_vocab=None, training=True, testing=False):
        # build dictionary that maps word/tag to a numeric value
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}

            for row in x:
                for token in row.split():
                    if token not in self.token_vocab:
                        # encode position when adding token into vocab (using length)
                        self.token_vocab[token] = len(self.token_vocab)
            for row in y:
                for token in row.split():
                    if token not in self.tag_vocab:
                        # encode position when adding tag into tag vocab (using length)
                        self.tag_vocab[token] = len(self.tag_vocab)
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab
        
        # convert sentences and tags to numbers using the dictionary
        self.corpus_token_ids = []
        self.corpus_tag_ids = []

        # for prediction, a placeholder to determine where padding is
        if testing:
            for row in x:
                tag_ids = [1 for token in row.split()]
                self.corpus_tag_ids.append(torch.tensor(tag_ids))

            for row in x:
                token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in row.split()]
                self.corpus_token_ids.append(torch.tensor(token_ids))

        # for training AND validation
        if testing == False:
            for row in x:
                token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in row.split()]
                self.corpus_token_ids.append(torch.tensor(token_ids))

            for row in y:
                tag_ids = [self.tag_vocab[tag] for tag in row.split()]
                self.corpus_tag_ids.append(torch.tensor(tag_ids))


    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]

# Model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embd_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        preds = self.fc(lstm_out)
        return preds
    
# Modify DataLoader's collate_fn to handle padded sequences
def padding(batch):
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]
    
    # set batch_first to True to make the batch size first dim
    padded_sentence = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab["<PAD>"])
    padded_tags = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab["<PAD>"])
    return padded_sentence, padded_tags

train_dataset = IOBDataset(x_train, y_train, training=True)
val_dataset = IOBDataset(x_val, y_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

vocab_size = len(train_dataset.token_vocab)
embd_size = 512
hidden_size = 256
output_size = len(train_dataset.tag_vocab)
num_layers = 2

model = BiLSTM(vocab_size, embd_size, hidden_size, output_size, num_layers)

pad_index = train_dataset.tag_vocab["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
optimizer = optim.AdamW(model.parameters(), lr=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=padding)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=padding)
model_path = os.path.join(os.path.dirname(__file__), "trained_model")

# Train model
prev_acc = None
prev_loss = None
num_epoch = 15
t0 = time.time()
for epoch in range(num_epoch):
    
    epoch_loss = 0
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        # changes shapes so that preds and y_batch are compatible on first dim
        loss = loss_fn(preds.view(-1, preds.shape[-1]), y_batch.view(-1))
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    scheduler.step(epoch_loss)

    model.eval()
    total_val_loss = 0
    all_predictions = []
    all_tags = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            outputs = model(x_val)
            outputs = outputs.view(-1, preds.shape[-1])
            tag_ids = y_val.view(-1)
            loss = loss_fn(outputs, tag_ids)
            total_val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            # select predictions and labels that aren't padding
            mask = tag_ids != train_dataset.tag_vocab['<PAD>']
            all_predictions.extend(predictions[mask].tolist())
            all_tags.extend(tag_ids[mask].tolist())

        val_loss = total_val_loss/ len(val_loader)
        val_acc = f1_score(all_tags, all_predictions, average="weighted")
        print(f"Epoch: {epoch + 1}/{num_epoch}, Validation Loss: {val_loss}, Validation Acc: {val_acc}")

        # save model with lowest loss and highest accuracy
        if prev_acc is None or (prev_acc <= val_acc and val_loss < prev_loss):
            print("HIGHEST ACC, LOWEST LOSS : SAVING MODEL")
            torch.save(model.state_dict(), model_path)
            prev_acc = val_acc
            prev_loss = val_loss

print(f"Training time on epoch {epoch}: {time.time()-t0}")

# Predict on Test Data
tag_lookup = {idx: tag for tag, idx in train_dataset.tag_vocab.items()}
test_df = os.path.join(os.path.dirname(__file__), "hw2_test.csv")
test_df = pd.read_csv(test_df)
results = os.path.join(os.path.dirname(__file__), "submission.csv")
test_data = IOBDataset(x=test_df["utterances"], y=None, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False, testing=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=padding)
saved_model = BiLSTM(vocab_size, embd_size, hidden_size, output_size, num_layers)
saved_model.load_state_dict(torch.load(model_path))
t0 = time.time()
saved_model.eval()
all_predictions = []
with torch.no_grad():
    for x_test, y_test in test_loader:
        sent_pred = []
        outputs = saved_model(x_test)
        outputs = outputs.view(-1, preds.shape[-1])
        tag_ids = y_test.view(-1)
        predictions = outputs.argmax(dim=1)
        mask = tag_ids != train_dataset.tag_vocab['<PAD>']
        for tag in predictions[mask]:
            sent_pred.append(tag_lookup[tag.item()])
        all_predictions.append(" ".join(sent_pred))

print(f"Predicting time on epoch {epoch}: {time.time()-t0}")

df = pd.DataFrame()
df["IOB Slot tags"] = all_predictions
df.index = df.index + 1
df.index.name = "ID"
df.to_csv(results)




