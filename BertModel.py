import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import AutoModel, BertTokenizerFast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layers
        self.fc1 = nn.Linear(768, 320)
        self.fc2 = nn.Linear(768, 160)
        self.fc3 = nn.Linear(768, 96)
        self.fc4 = nn.Linear(768, 32)

        # output layer
        self.fc5 = nn.Linear(32, 6)

        #a apply softmax
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):

        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)
        x = self.fc2(cls_hs)
        x = self.fc3(cls_hs)
        x = self.fc4(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc5(x)

        x = self.softmax(x)

        return x

def apply_bert(data):

    data["xOTIOverall"] = data["xOTIOverall"].astype(int)

    train_text, temp_text, train_labels, temp_labels = train_test_split(data["Mission Statement"],
                                                                        data["xOTIOverall"],
                                                                        random_state=42,
                                                                        test_size=0.3,
                                                                        stratify=data["xOTIOverall"])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=42,
                                                                    test_size=0.5, stratify=temp_labels)

    bert = AutoModel.from_pretrained("bert-base-uncased")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=512,
        pad_to_max_length=True,
        truncation=True
    )

    ## convert lists to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    train_y -= 1

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    val_y -= 1

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    batch_size = 32

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sample the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # as above
    val_data = TensorDataset(val_seq, val_mask, val_y)

    val_sampler = SequentialSampler(val_data)

    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # pass BERT to architecture
    model = BERT_Arch(bert)

    # push the model to device
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=1e-5)  # learning rate

    # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

    print("Class Weights:", class_weights)

    # converting class weights to tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    epochs = 200

    def train():

        model.train()

        total_loss, accuracy = 0, 0

        # empty list to save model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(train_dataloader):

            # progress update after every batch
            if step % 1 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            batch = [r.to(device) for r in batch]

            sent_id, mask, labels = batch

            # clear previously calculated gradients
            model.zero_grad()

            # get model predictions for the current batch
            preds = model(sent_id, mask)

            # compute the loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            # add on to the total loss
            total_loss = total_loss + loss.item()

            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0 to combat exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # push predictions to CPU if gpu present
            preds = preds.detach().cpu().numpy()

            # append the model predictions
            total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def evaluate():

        # mirrors above

        print("\nEvaluating...")

        model.eval()

        total_loss, total_accuracy = 0, 0

        total_preds = []

        for step, batch in enumerate(val_dataloader):

            if step % 1 == 0 and not step == 0:

                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")

    if not os.path.isdir("./metrics/bert"):
        os.mkdir("./metrics/bert")

    output = str(model)

    model_file = open(f"./metrics/bert/model-info.txt", "w")

    model_file.write(output)

    model_file.close()

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train()

        # evaluate model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "./metrics/bert/saved_weights.pt")

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"\nTraining Loss: {train_loss:.3f}")
        print(f"Validation Loss: {valid_loss:.3f}")

    print(f"\nGenerating best metrics from saved model:")
    path = "./metrics/bert/saved_weights.pt"
    model.load_state_dict(torch.load(path))

    file = open(f"./metrics/bert/classification-report-{epochs}-epochs.txt", "w")

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))

        preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis=1)
        print(classification_report(test_y, preds))
        file.write(classification_report(test_y, preds))
        file.close()


