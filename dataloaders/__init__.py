from torch.utils.data import DataLoader
from dataloaders.video_list import get_loader, test_dataset

# dataloader for video COD
def video_dataloader(args):          
    train_loader = get_loader(dataset_root=args.dataset_root,
                              batchsize=args.batchsize,
                              trainsize=args.trainsize,
                              train_split=args.trainsplit,
                              num_workers=8,
                              )
    val_loader = test_dataset(dataset_root=args.dataset_root,
                              split=args.testsplit,
                              testsize=args.trainsize,
                              )
    print('Training with %d image pairs' % len(train_loader))
    print('Val with %d image pairs' % len(val_loader))
    return train_loader, val_loader 


def test_dataloader(args):   
    test_loader = test_dataset(dataset_root=args.dataset_root,
                              split=args.testsplit,
                              testsize=args.testsize)
    print('Test with %d image pairs' % len(test_loader))
    return test_loader 