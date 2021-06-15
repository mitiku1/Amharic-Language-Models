import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=10, type = int, helper="Number of epochs to train the model")
    parser.add_argument("--lr",  default=0.1, type = float, helper="Learning rate to use for training")
    parser.add_argument("--batch-size",  default=128, type = float, helper="DataLoader batch size")
    
    args = parser.parse_args()
    
    return args