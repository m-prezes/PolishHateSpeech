import sys
from pathlib import Path
import pandas as pd
import yaml
from evaluate import evaluate
from tqdm import tqdm

from model import Model
import torch

from dataset import PolishHateSpeechDataset

from torch.utils.data import DataLoader

from tqdm import tqdm


def train(model, train_data_loader, val_data_loader, compute_loss, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for i, (data, targets) in tqdm(enumerate(train_data_loader), f"Epoch {epoch+1}/{num_epochs}",
                                       total=len(train_data_loader)):
            data['input_ids'] = data['input_ids'].squeeze(1).to(device)
            data['attention_mask'] = data['attention_mask'].squeeze(1).to(device)
            # data['token_type_ids'] = data['token_type_ids'].squeeze(1).to(device)

            targets = targets.to(device)
            model.train()
            optimizer.zero_grad()
        
            outputs = model(data)
            print(torch.cat((outputs, targets.view(-1, 1)), 1))

            loss = compute_loss(outputs, targets.view(-1, 1).to(torch.float32))

            loss.backward()
            optimizer.step()


        # val_loss, val_acc, val_gmean = evaluate(model, val_data_loader, compute_loss, device)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}, Validation Acc: {val_acc}, Validation GMean: {val_gmean}")

    torch.save(model.state_dict(), 'model.pth')
    

    return model




if __name__=="__main__":
    params = yaml.safe_load(open('params.yaml'))['train']
    epochs = params['epochs']
    lr = params['lr']
    batch_size = params['batch_size']

    X_train = pd.read_csv(Path(sys.argv[1]), header=None)
    y_train = pd.read_csv(Path(sys.argv[2]), header=None)
    X_val = pd.read_csv(Path(sys.argv[3]), header=None)
    y_val = pd.read_csv(Path(sys.argv[4]), header=None)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.to(device)
    tokenizer = model.tokenizer

    train_dataset = PolishHateSpeechDataset(X_train, y_train, tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PolishHateSpeechDataset(X_val, y_val, tokenizer)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    compute_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_data_loader, val_data_loader, compute_loss, optimizer, epochs, device)