import torch 



def get_label_matches(l1,l2, d, l):
    lmask = torch.where(torch.logical_or(l==l1, l==l2), 1,0)
    #print(lmask)
    indices = torch.nonzero(lmask).squeeze()
    print(indices)
    # unsqueeze for the channel dimension here. 
    return d[indices].unsqueeze(1), l[indices]

def split_dataset():
    train_data, train_labels = torch.load('../data/MNIST/processed/training.pt')
    print("Loaded train in")
    test_data, test_labels = torch.load('../data/MNIST/processed/test.pt')
    print("Loaded train and test in")

    for i in range(5):
        print(i)
        l1 = i*2
        l2 = l1+1

        train_d, train_l = get_label_matches(l1,l2, train_data, train_labels)
        test_d, test_l = get_label_matches(l1,l2, test_data, test_labels)

        print(train_d.shape, test_d.shape, len(train_l), len(test_l), train_l.shape)

        torch.save((train_d, train_l), "../data/splits/MNIST/split_train_"+str(i)+".pt")
        torch.save((test_d, test_l), "../data/splits/MNIST/split_test_"+str(i)+".pt")


if __name__ == '__main__':
    split_dataset() 