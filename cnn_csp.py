# This is a sample Python script.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from random_loader import RolloutObservationDataset
import argparse
import sys

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class img_cnn2(nn.Module):
    def __init__(self):
        super(img_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(257,2)
        self.sft=nn.Softmax()

    def forward(self, x,a):
        input_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # a=a.view(128,3)
        x=torch.cat((x,a.unsqueeze(1)),1)
        x = self.fc3(x)
        x=self.sft(x)

        return x

class img_cnn(nn.Module):
    def __init__(self):
        super(img_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(259,2)
        self.sft=nn.Softmax()

    def forward(self, x,a):
        input_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # a=a.view(128,3)
        x=torch.cat((x,a),1)
        x = self.fc3(x)
        x=self.sft(x)

        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入通道数3，输出通道数16，卷积核大小3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 第二个卷积层：输入通道数16，输出通道数32，卷积核大小3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 第三个卷积层：输入通道数32，输出通道数64，卷积核大小3x3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 池化层：2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 经过第一个卷积层和池化层
        x = self.pool(F.relu(self.conv2(x)))  # 经过第二个卷积层和池化层
        x = self.pool(F.relu(self.conv3(x)))  # 经过第三个卷积层和池化层
        x = x.view(-1, 64 * 28 * 28)  # 展平操作
        x = F.relu(self.fc1(x))  # 经过第一个全连接层
        x = self.fc2(x)  # 经过第二个全连接层
        return x
def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def main(train_path,test_path,step_save_name,step,epochs,task):
    if task==1:
        rnn = SimpleCNN()
    elif task==2:
        rnn=SimpleCNN()
    else:
        print("Invalid task number! `", sys.exc_info()[0])

    cur_best = None
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    # Training loop
    num_epochs = epochs

    train_dataset = RolloutObservationDataset(train_path, leng=step)
    test_dataset = RolloutObservationDataset(test_path, leng=step)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    rnn = rnn.to(device)
    train_loss = []
    test_loss = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    for epoch in range(num_epochs):
        correct=0
        total_loss = 0
        total_num = 0
        pbar = tqdm(total=len(train_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        rnn.train()
        for data in train_loader:  # Iterate over your training dataset
            inputs, safes = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            safes=safes.to(device)
            obs = inputs.float()
            # zs=vae(inputs)
            zoutputs = rnn(obs)
            _, predicted = torch.max(zoutputs.data, 1)
            correct += (predicted == safes).sum().item()
            # Forward pass

            # # Compute loss
            # print(zoutputs.shape, zoutputs.dtype)
            # print(safes.shape, safes.dtype)
            loss = criterion(zoutputs.float(), safes.to(device).long())
            total_loss = total_loss + loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        # Print the epoch loss
        print(f"Training Epoch [{epoch + 1}/{num_epochs}], Acc:{correct/total_num}  Loss: {total_loss / total_num:.10f}")
        wandb.log({ "accuracy":  correct/total_num})

        train_loss.append(total_loss / total_num)
        pbar.close()
        correct=0

        rnn.eval()
        total_loss = 0
        total_num = 0
        test_preds = []
        test_trues = []
        pbar = tqdm(total=len(test_loader),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        for data in test_loader:  # Iterate over your training dataset
            inputs, safes = data
            total_num = total_num + len(inputs)
            inputs = inputs.to(device)
            obs = inputs.float()
            safes=safes.to(device)

            # zs=vae(inputs)
            zoutputs = rnn(obs)
            _, predicted = torch.max(zoutputs.data, 1)
            correct += (predicted == safes).sum().item()
            test_preds.extend(predicted.detach().cpu().numpy())
            test_trues.extend(safes.detach().cpu().numpy())
            # Forward pass

            # Compute loss
            loss = criterion(zoutputs.float(), safes.to(device).long())
            total_loss = total_loss + loss.item()

            pbar.update(1)

        print(f"Test Epoch [{epoch + 1}/{num_epochs}],Acc:{correct/total_num}  Loss: {total_loss / total_num:.10f}")
        test_loss.append(total_loss / total_num)
        pbar.close()
        val_loss = total_loss / total_num
        scheduler.step(val_loss)
        conf_matrix = confusion_matrix(test_trues, test_preds)
        print(conf_matrix)
        f1 = f1_score(test_trues, test_preds)
        print(f"F1 Score: {f1:.2f}")

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(test_trues, test_preds).ravel()

        # 计算FPR (False Positive Rate)
        fpr = fp / (fp + tn)
        print(f"False Positive Rate (FPR): {fpr:.2f}")
        is_best = not cur_best or val_loss < cur_best
        if is_best:
            cur_best = val_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': rnn.state_dict(),
            'train_loss': train_loss,
            'test_loss': val_loss,
            'f1':f1,
            'fpr':fpr,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, step_save_name+'-best.tar', step_save_name+'-checkpooint.tar')

if __name__ == '__main__':

    device="cpu"
    print(device)
    batchsize=128
    parser = argparse.ArgumentParser(description='predictor trainer')
    parser.add_argument('--train', default="./train",
                      )
    parser.add_argument('--test',default="./test",
                      )
    parser.add_argument('--save',default=".\\",
                      )
    parser.add_argument('--epochs',default=101,
                       )
    parser.add_argument('--steps',default=10,
                      )
    parser.add_argument('--task',default=1,
                      )

    args = parser.parse_args()


    train_path=args.train
    test_path=args.test
    save_path=args.save
    steps = int(args.steps)
    epochs=int(args.epochs)
    task=int(args.task)
    import wandb
    # 1. shuffle the data, ratio of training and test 4:1
    # 2. sign up an account of wandb, send me the username. I will invite you to the org.
    # 3. add the training and test metrics: acc, loss, f1 score, fpr
    #        wandb.log({ "accuracy":  loss2sum / (len(TrainLoader)*batch_size)})
    

    for i in range(1,steps):
        wandb.init(
            # set the wandb project where this run will be logged
            project="interval-easy-dvae",

            # track hyperparameters and run metadata
            config={
                "step": i,

            }
        )
        main(train_path,test_path,save_path+"step"+str(i),i*3,epochs,task)



