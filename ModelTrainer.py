import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, corpus, batch_size=24, learning_rate=0.01, epochs=10, device=None):
        self.model = model
        self.corpus = corpus
        self.vocab_size = corpus.vocab_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")        
        self.criterion = nn.CrossEntropyLoss(ignore_index=corpus.vocab["<pad>"])
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
    
    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            dec_inputs, target_batch = self.corpus.make_batch(self.batch_size)
            dec_inputs, target_batch = dec_inputs.to(self.device), target_batch.to(self.device)
            outputs = self.model(dec_inputs)
            loss = self.criterion(outputs.view(-1, self.corpus.vocab_size), target_batch.view(-1))
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
            loss.backward()
            self.optimizer.step()