from model import SequenceTagger
from dataset import loader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from utils import AccuracyCounter, logging
from operator import itemgetter
import torch

def train(model, loss_func, epochs, optimizer, train_loader, eval_loader, device=None):
    device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt_str = str(optimizer).replace('\n ', ',')
    logging(f'Training - loss:{loss_func}, epochs:{epochs}, optimizer:{opt_str}, device:{device}')
    for epoch in range(epochs):
        # Train
        model.train()
        avg_loss = None
        train_accuracy = AccuracyCounter()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            out = model(data)
            loss = loss_func(out, target.to(device))
            avg_loss = loss.item() if avg_loss is None else (0.99*avg_loss + 0.01*loss.item())
            train_accuracy.compute_from_soft(out, target.to(device))
            loss.backward()
            optimizer.step()
        train_accuracy_val = train_accuracy.get_accuracy_and_reset()
        # Eval
        model.eval()
        with torch.no_grad():
            eval_accuracy = AccuracyCounter()
            for data, target in eval_loader:
                out = model(data)
                eval_accuracy.compute_from_soft(out, target.to(device))
            eval_accuracy_val = eval_accuracy.get_accuracy_and_reset()
            logging('Done epoch {}/{} ({} batches) train accuracy {:.2f}, eval accuracy {:.2f} avg loss {:.5f}'.format(
                epoch+1, epochs, (epoch+1)*train_loader.__len__(), train_accuracy_val, eval_accuracy_val, avg_loss))




if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SequenceTagger(126, 50, 50, 50, 3).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    train(model, CrossEntropyLoss(), 3, optimizer, loader('data/train', 50), loader('data/test', 500))