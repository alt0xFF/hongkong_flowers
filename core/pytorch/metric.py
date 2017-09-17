import torch

def simple_metric(outputs, labels):
    # predicted is the indices with maximum probability.
    _, predicted = torch.max(outputs, 1)
    total = len(labels)
    correct = (predicted == labels).sum().float()
    accuracy = correct / total
    
    return accuracy.data.cpu().numpy()