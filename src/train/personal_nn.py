import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import os

# trained on name, gender, dob, and job
class PersonalLinearNet(nn.Module):
    def __init__(self, num_names: int, num_jobs: int, embedding_dim: int):
        super(PersonalLinearNet, self).__init__()
        self.name_embedding = nn.Embedding(num_names, embedding_dim)
        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        
        input_dim = embedding_dim * 2 + 1 + 3
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, names, jobs, gender, dob):
        name_input = self.name_embedding(names)
        job_input  = self.job_embedding(jobs)
        x = torch.cat([name_input, job_input, gender.unsqueeze(1), dob], dim=1)
        return self.model(x)

class PersonalDropoutNet(nn.Module):
    pass

def train(embedding_dim:int , epochs: int, batch_size: int, lr: float, weight_decay: float):
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Processed", "10mil", "personal_10000000.csv")
    df = pd.read_csv(train_data_path)

    name_to_idx = {name: idx for idx, name in enumerate(df["name"].unique())}
    job_to_idx = {job: idx for idx, job in enumerate(df["job"].unique())}
    genders = torch.tensor([0 if gender == "F" else 1 for gender in df["gender"]], dtype=torch.float)
    dobs = torch.tensor([[int(x) for x in dob.split("-")] for dob in df['dob']], dtype=torch.float)
    labels = torch.tensor([x for x in df["is_fraud"]], dtype=torch.float)
    name_indices = torch.tensor([name_to_idx[name] for name in df["name"]], dtype=torch.long)
    job_indices = torch.tensor([job_to_idx[job] for job in df["job"]], dtype=torch.long)

    max_dob = 2008
    min_dob = 1928
    dobs[:, 0] = (dobs[:, 0] - min_dob) / (max_dob - min_dob)
    dobs[:, 1] = (dobs[:, 1] - 1) / 11
    dobs[:, 2] = (dobs[:, 2] - 1) / 30


    model = PersonalLinearNet(len(name_to_idx), len(job_to_idx), embedding_dim)
    dataset = TensorDataset(name_indices, job_indices, genders, dobs, labels)
    data = DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        for bidx, (name_indices, job_indices, genders, dobs, y) in enumerate(data):
            optimizer.zero_grad()
            pred = model(name_indices, job_indices, genders, dobs)
            loss = criterion(pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            if (bidx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{bidx+1}/{len(data)}], Loss: {loss.item():.4f}')

if __name__ == "__main__":

    embedding_dim = 25
    epochs = 100
    batch_size = 1024
    lr = .001
    weight_decay = .01

    train(embedding_dim, epochs, batch_size, lr, weight_decay)
    #print(torch.cuda.is_available())