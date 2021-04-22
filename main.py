import torch
import pandas as pd
from dataset import MyDataset, Rescale, Crop, ToTensor
from unet import UNet
from torchvision import transforms
import copy
from skimage import io
import re

def criterion(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


trans_raw = transforms.Compose([
    Rescale(572),
    Crop(572),
    ToTensor()
])

trans_proc = transforms.Compose([
    Rescale(484),
    Crop(484),
    ToTensor()
])

raw_train_dir = "./raw_photos/train/"
processed_train_dir = "./processed_photos/train/"
raw_val_dir = "./raw_photos/val/"
processed_val_dir = "./processed_photos/val/"

train_dataset = MyDataset(raw_train_dir, processed_train_dir, trans_raw, trans_proc)
val_dataset = MyDataset(raw_val_dir, processed_val_dir, trans_raw, trans_proc)
my_datasets = {"train": train_dataset, "val": val_dataset}

cnn_model = UNet(1,3)

optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.99)

def train(datasets, num_epoch=30):
    best_model_wts = copy.deepcopy(cnn_model.state_dict())
    best_loss = 1.0
    loss_history = {'train': [], 'val': []}

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn_model.train()  # Set model to training mode
            else:
                cnn_model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            dataset_ = datasets[phase]
            for idx in range(len(dataset_)):
                inputs, labels = dataset_[idx].values()
                inputs = inputs.unsqueeze(0)
                labels = labels.unsqueeze(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn_model(inputs)
                    outputs = outputs.permute(0, 2, 3, 1)
                    labels = labels.resize(484*484*3)
                    outputs = outputs.resize(484*484*3)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                epoch_loss = running_loss / len(datasets[phase])
                loss_history['train'].append(epoch_loss)
            elif phase == 'val':
                epoch_loss = running_loss / len(datasets[phase])
                loss_history['val'].append(epoch_loss)

                print('Loss: {:.4f}'.format(epoch_loss))

                # deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(cnn_model.state_dict())

    print('Best Loss: {:4f}'.format(best_loss))
    cnn_model.load_state_dict(best_model_wts)

    pd.DataFrame(data=loss_history).to_csv(f"./loss.csv")
    torch.save(cnn_model.state_dict(), f"./model.pt")

def predict(dataset):
    cnn_model.load_state_dict(torch.load(f"./model.pt"))
    was_training = cnn_model.training
    cnn_model.eval()
    for idx in range(len(dataset)):
        inputs, _ = dataset[idx].values()
        inputs = inputs.unsqueeze(0)

        outputs = cnn_model(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        filename = dataset.proc_dataset[idx]
        temp = re.search(r"\w*\.\w*$", filename)
        filename = temp.group(0)
        filename = f"./predicted_photos/{filename}"
        outputs = outputs.squeeze(0).detach().numpy()
        io.imsave(filename, outputs)

    cnn_model.train(mode=was_training)


train(my_datasets)
predict(my_datasets['val'])


