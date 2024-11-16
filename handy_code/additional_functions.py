from torch.utils.data import Dataset 
class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.x[idx]

        # apply transforms if provided
        if self.transform:
            x = self.transform(x)

        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

# shows percentage of class labels in mini-batches across the dataset
def count_freq_in_batch(data_loader,labels,data_len):
    from collections import Counter as cnt
    for label in labels:
        it = iter(data_loader)
        s = sum([cnt((list(t[1].squeeze().numpy())))[label] for t in it])
        print('count of label %.1f = %.2f' %(label,(s)))
        print('freq of label %.1f = %.2f' %(label,(s/data_len)*100))
