import torch
import torch.nn as nn
import torch.optim as optim
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
            nn.Linear(4, 1)
        )

    def forward(self, name_idx, job_idx, gender, dob):
        name_input = self.name_embedding(name_idx)
        job_input  = self.name_embedding(job_idx)
        
        x = torch.cat([name_input, job_input, gender, dob.unsqueeze(1)], dim=1)
        
        return self.model(x)

class PersonalDropoutNet(nn.Module):
    pass

def train():
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Processed", "10mil", "personal_10000000.csv")
    df = pd.read_csv(train_data_path)
    print(df.head())




if __name__ == "__main__":
    train()