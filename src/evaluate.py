from pathlib import Path
import sys
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import yaml

from model import Model

from dataset import PolishHateSpeechDataset

from torch.utils.data import DataLoader

def evaluate(model, data_loader, compute_loss, device):
    model.eval()

    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, targets in tqdm(data_loader, f"Evaluate", total=len(data_loader)):
            data['input_ids'] = data['input_ids'].squeeze(1).to(device)
            data['attention_mask'] = data['attention_mask'].squeeze(1).to(device)
            data['token_type_ids'] = data['token_type_ids'].squeeze(1).to(device)

            targets = targets.to(device)
            targets = targets.view(-1, 1).to(torch.float32)

            outputs = model(data)

            predicted = torch.round(outputs)
            loss = compute_loss(outputs, targets)
            total_correct += (predicted == targets).sum().item()
            
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


if __name__=="__main__":
    params = yaml.safe_load(open('params.yaml'))['evaluate']
    batch_size = params['batch_size']

    test_texts = pd.read_fwf(Path(sys.argv[1]), header=None)
    test_labels = pd.read_fwf(Path(sys.argv[2]), header=None)

    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataset = PolishHateSpeechDataset(test_texts, test_labels, model.tokenizer)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    compute_loss = torch.nn.BCELoss()

    loss, acc = evaluate(model, test_data_loader, compute_loss, device)
    print(f"Test Loss: {loss}, Test Acc: {acc}")