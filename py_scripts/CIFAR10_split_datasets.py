import torch 
import numpy as np
from PIL import Image
from torchvision import transforms

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_label_matches(l1,l2, d, l):
    lmask = torch.where(torch.logical_or(l==l1, l==l2), 1,0)
    indices = lmask.nonzero().squeeze()
    print(indices )
    return d[indices], l[indices].type(torch.int64)


def data_processing(data, labels):
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    trans = transforms.ToTensor()
    # going from PIL to torch tensor converts things to floats as desired. 
    data = torch.stack([trans(Image.fromarray(d)) for d in data])
    labels = torch.Tensor(labels)
    return data, labels

def split_dataset():
    data, labels = None, []
    # unify data from all of the different batches. 
    for i in range(5):
        t_dict = unpickle('../data/cifar-10-batches-py/data_batch_'+str(i+1))
        if data is None: 
            data = t_dict[b'data']
        else: 
            data = np.concatenate((data, t_dict[b'data']))
        labels += t_dict[b'labels']

    data, labels = data_processing(data, labels)

    # get the test data. 
    t_dict = unpickle('../data/cifar-10-batches-py/test_batch')
    t_data, t_labels = t_dict[b'data'], t_dict[b'labels']
    t_data, t_labels = data_processing(t_data, t_labels)

    print(data.shape, t_data.shape, len(labels), len(t_labels), labels.shape )

    for i in range(5):
        l1 = i*2
        l2 = l1+1

        train_d, train_l = get_label_matches(l1,l2, data, labels)
        test_d, test_l = get_label_matches(l1,l2, t_data, t_labels)

        print(train_d.shape, test_d.shape, len(train_l), len(test_l), train_l.shape)

        torch.save((train_d, train_l), "../data/splits/CIFAR10/split_train_"+str(i)+".pt")
        torch.save((test_d, test_l), "../data/splits/CIFAR10/split_test_"+str(i)+".pt")

if __name__ == '__main__':
    split_dataset() 

