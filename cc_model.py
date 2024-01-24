# Le Thanh Hai
# June 07, 2023
# Load 36 polarized images in color, but will take red channel from 36 images
# stack together to make an input of CxHxW with C be 36, not 3 channels

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time
import copy
import json
from tqdm import tqdm
from timeit import default_timer as timer

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler
from torchmetrics import F1Score

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

from alb_transform import prepare_transform
from dataloader_class import get_loaders, get_test_loaders
from utils import EarlyStopping, LRScheduler, write_data
from build_model_2 import CNN, CNN_2, CNN_2_2
from pretrained_models import load_densenet121, load_efficientnet

# ------ Initial parameters for training model
root = Path('../CC_ViHiep')
image_dir = root / 'data_small'
list_file_path = root / 'lists'
log_dir = root / 'logs'
result_path = log_dir / 'result'

batch_size = 16  # 8, 16, 32
n_epochs = 200
n_classes = 1  # binary classification
class_names = ['Normal', 'Colorectal']
"""
The labels in the provided data are: 0 for Normal sample, 1 for Colorectal cancer
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(pretrained=False):
    torch.cuda.empty_cache()
    print('Start training a model')
    train_path = list_file_path / 'train.txt'
    valid_path = list_file_path / 'valid.txt'
    
    print(f'torch: {torch.__version__}')

    # train_transforms, valid_transforms, show_transforms = prepare_transform()
    train_transforms, valid_transforms, _ = prepare_transform()
    print(f'batch_size: {batch_size}')

    device = torch.cuda.current_device()
    print(f'device: {device}')

    loaders = get_loaders(image_dir=image_dir,
                          train_path=train_path,
                          valid_path=valid_path,
                          batch_size=batch_size,
                          train_transforms_fn=train_transforms,
                          valid_transforms_fn=valid_transforms,
                          )
           
    # pretrained = True for Densetnet, Efficientnet
    # pretrained = False for training models from scratch
    if pretrained:        
        model = load_densenet121(in_channels=36, num_classes=n_classes)
        # model = load_efficientnet(in_channels=36, num_classes=n_classes)
        learning_rate = 1e-3  # 1e-5, 1e-3
        weight_decay = 1e-3  # 3e-4, 1e-2
    else:
        # model = CNN(input_channels=36, num_classes=n_classes)        
        model = CNN_2_2(input_channels=36, num_classes=n_classes)  # architecture is same as CNN but uses only Conv layer
        learning_rate = 1e-2  # 1e-5, 1e-3, 1e-2
        weight_decay = 1e-3  # 3e-4, 1e-2, 1e-3, 1e-4
    print(model)
    
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 3e-4=0.0003
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    # LRScheduler class uses ReduceLROnPlateau
    scheduler = LRScheduler(optimizer=optimizer, patience=10, min_lr=1e-6, factor=0.1)
    
    # loss function    
    criterion = nn.BCEWithLogitsLoss()

    # metric function
    metric = F1Score(task='binary', num_classes=n_classes, average='macro')

    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping(patience=20, min_delta=0)    

    # The sanity_check flag breaks a training epoch after one mini-batch,
    # meaning the loops are executed more quickly.
    params_train = {
        'num_epochs': n_epochs,
        'optimizer': optimizer,
        'loss_func': criterion.to(device),
        'train': loaders['train'],
        'val': loaders['valid'],
        'sanity_check': False,  # True, False
        'lr_scheduler': scheduler,
        'early_stopping': early_stopping,
        'device': device,
        'path2weights': log_dir / 'checkpoints',
        'metric_func': metric.to(device)
    }

    # train and validate the model
    model, loss_history, metric_history = train_val(model.to(device), params_train)

    # evaluate the trained model with validation set
    test_trained_model(model=model, fname=valid_path)

    # test the trained model with testing set (unseen data)
    test_trained_model(model=model, fname='test.txt')   

    torch.cuda.empty_cache()


def train_val(model, params):
    # extract model parameters
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_ds = params['train']
    val_ds = params['val']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    early_stopping = params['early_stopping']
    device = params['device']
    path2weights = params['path2weights']
    metric_func = params['metric_func']

    if not path2weights.exists():
        print(f'Directory {path2weights} does not exist. Make new one.')
        Path.mkdir(path2weights)

    # history of loss values in each epoch
    loss_history = {
        'train': [],
        'val': [],
    }

    # history of metric values in each epoch
    metric_history = {
        'train': [],
        'val': [],
    }

    # history of best loss, best accuracy at best epoch
    info_history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
    }

    # a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float('inf')
    best_epoch = 0
    best_score = 0

    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = lr_scheduler.get_lr()       
        print('Epoch {}/{}, current lr={}'.format(epoch + 1, num_epochs, current_lr))

        # train model on training dataset
        print('Training')
        model.train()
        train_loss, train_metric = loss_epoch(model=model, loss_func=loss_func, dataset=train_ds,
                                              sanity_check=sanity_check, opt=opt, device=device,
                                              epoch=epoch + 1, metric_func=metric_func
                                              )

        # collect loss and metric for training dataset
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        # evaluate model on validation dataset
        print('Validating')
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model=model, loss_func=loss_func, dataset=val_ds,
                                              sanity_check=sanity_check, opt=None, device=device,
                                              epoch=epoch + 1, metric_func=metric_func
                                              )
            # print(val_metric)

        # store best model
        if val_loss < best_loss:
        # if best_score < val_metric:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_score = val_metric
            info_history['epoch'].append(best_epoch)
            info_history['learning_rate'].append(current_lr)
            info_history['train_loss'].append(train_loss)
            info_history['val_loss'].append(best_loss)
            info_history['train_accuracy'].append(train_metric)
            info_history['val_accuracy'].append(best_score)

            # store weights into a local file
            torch.save(model.state_dict(), path2weights / 'best.pth')            
            print("Copied best model weights!")

        # collect loss and metric for validation dataset
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        # learning rate schedule
        # If the validation loss does not decrease for the given number of 'patience' epochs,
        # then the learning rate will decrease by given 'factor'.
        lr_scheduler(val_loss)  # val_loss, 1., val_metric
        if current_lr != lr_scheduler.get_lr():
            print("LR decreased ==> Loading the best model weights!")
            model.load_state_dict(best_model_wts)
        
        print("train loss: %.6f, val. loss: %.6f, accuracy: %.3f" % (train_loss, val_loss, val_metric))

        if early_stopping is not None:
            early_stopping(val_loss=val_loss)
            if early_stopping.early_stop:
                print('Early stopping breaks the progress!')
                break

        print("-" * 40)

    # load best model weights
    model.load_state_dict(best_model_wts)    
    print('Best loss: {0:.3f} and best accuracy: {1:.3f} with epoch {2:3d}'.format(best_loss, best_score, best_epoch))

    save_json(info_history, loss_history, metric_history, path2weights, model.__class__.__name__)

    return model, loss_history, metric_history


def loss_epoch(model, loss_func, dataset, sanity_check=False, opt=None, device='cuda', epoch=0, metric_func=None):
    running_loss = 0.0
    running_metric = 0.0    
    n_metric = 0
    n_loss = 0

    # use tqdm to draw a progress bar of an epoch when training or validating model
    with tqdm(dataset, unit='batch') as tepoch:
        # for xb, yb in dataset:
        for i, features in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            # move batch to device
            xb = features['images'].to(device)
            yb = features['targets'].to(device)

            # get model output
            output = model(xb)

            # get loss per batch
            loss_b, metric_b = loss_batch(loss_func, output, yb, opt, metric_func)

            # update running loss
            running_loss += loss_b
            n_loss += 1

            # update running metric
            if metric_b is not None:
                running_metric += metric_b
                n_metric += 1

            # break the loop in case of sanity check
            if sanity_check is True:
                break

            tepoch.set_postfix(loss=loss_b, accuracy=metric_b)
            time.sleep(0.1)

    # average loss value    
    loss = running_loss / float(n_loss)

    # average metric value    
    metric = running_metric / float(n_metric)

    return loss, metric


def loss_batch(loss_func, output, target, opt=None, metric_func=None):
    # get loss
    # because of using CrossEntropyLoss so do not need to apply softmax function to output    
    loss = loss_func(output, target)
    
    # get performance metric
    metric_b = metrics_batch(output, target, metric_func)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def metrics_batch(output, target, metric_func=None):
    # get output class
    # because the output of model does not apply an activation function,
    # so need to apply activation function for output, softmax/sigmoid
    pred = torch.sigmoid(output)
    targets = target.type(torch.int64)  # convert float tensor to int tensor for metric function from torchmetrics

    if metric_func is not None:
        accuracy = metric_func(pred, targets)
        accuracy = accuracy.cpu().numpy()
    else:
        # convert gpu tensor to numpy array
        pred_batch = pred.cpu().numpy()
        target_batch = target.cpu().numpy()        
        accuracy = f1_score(target_batch, pred_batch, average='weighted')
     
    return accuracy


# save results to json file
def save_json(best_epoch_info, loss_info, metrics_info, path2dic, model_name='CNN'):
    # write information about any best epoch, best loss, and best accuracy
    # Serializing json
    info_obj = json.dumps(best_epoch_info, indent=6)
    loss_obj = json.dumps(loss_info, indent=3)
    metric_obj = json.dumps(metrics_info, indent=3)    

    # Writing to sample.json
    with open(path2dic / 'metrics.json', 'w') as outfile:
        outfile.write(json.dumps(f'Model is {model_name}', indent=1))
        outfile.write('\n')
        outfile.write(json.dumps('Best: ', indent=1))
        outfile.write(info_obj)
        outfile.write('\n')
        outfile.write(json.dumps('Loss: ', indent=1))
        outfile.write(loss_obj)
        outfile.write('\n')
        outfile.write(json.dumps('Accuracy: ', indent=1))
        outfile.write(metric_obj)
    print('Saved the values of loss and accuracy per epoch.')


def test_trained_model(model=None, fname='test.txt'):
    # torch.cuda.empty_cache()
    start = timer()
    print('Start testing the trained model')
    if Path(fname).exists():
        test_path = fname
    else:
        test_path = list_file_path / fname
    print(f'Data set is {test_path}')

    _, valid_transforms, _ = prepare_transform()

    loaders = get_test_loaders(image_dir=image_dir, test_path=test_path,
                               batch_size=batch_size, test_transforms_fn=valid_transforms
                               )
    if model is not None:
        model.to('cpu')
    else:
        # load the weights from file
        # model = CNN(input_channels=36, num_classes=n_classes)
        # model = CNN_2(input_channels=36, num_classes=n_classes)
        # model = CNN_2_2(input_channels=36, num_classes=n_classes)        
        model = load_densenet121(in_channels=36, num_classes=n_classes)
        # model = load_efficientnet(in_channels=36, num_classes=n_classes)

        model.load_state_dict(torch.load(log_dir / 'checkpoints' / 'best.pth'))        
    
    model.eval()

    threshold = 0.5
    test_ds = loaders['test']
    n_tests = len(test_ds.dataset)
    print(f'No of test samples: {n_tests:3d}')    
    for i, features in enumerate(test_ds):
        pred_prob = torch.sigmoid(model(features['images']))        
        pred = (pred_prob > threshold).float()  # pred_prob.round()        
        target = features['targets']  # binary classification, 0/1        
        fn_batch = features['filename']
        
        # convert cpu torch tensor to numpy array
        pred_batch = pred.numpy()
        target_batch = target.numpy()
        pred_prob_batch = pred_prob.detach().numpy()
        target_oh_batch = target.numpy()        
        if i == 0:
            y_preds = pred_batch
            y_trues = target_batch
            y_preds_prob = pred_prob_batch
            y_trues_oh = target_oh_batch
            filenames = fn_batch
        else:
            y_preds = np.vstack((y_preds, pred_batch))
            y_trues = np.vstack((y_trues, target_batch))
            y_preds_prob = np.vstack((y_preds_prob, pred_prob_batch))
            y_trues_oh = np.vstack((y_trues_oh, target_oh_batch))
            filenames += fn_batch            

    # check which labels are not predicted by model
    print(np.setxor1d(y_trues, y_preds))    

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_trues, y_preds)
    print('Accuracy: %f' % accuracy)

    # if use micro, macro, weighted, all metrics are same (average value)
    # for binary classification, use 'binary'
    precision, recall, f1, _ = precision_recall_fscore_support(y_trues, y_preds,
                                                               average='binary',
                                                               zero_division=1)  # labels=np.unique(y_preds)

    # precision tp / (tp + fp)    
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)    
    print('Recall: %f' % recall)
    # f1: 2*tp / (2*tp + fp + fn)    
    print('F1 score: %f' % f1)

    auc_score = roc_auc_score(y_trues_oh, y_preds_prob, average='macro')
    print('AUC score: %f' % auc_score)

    matrix = confusion_matrix(y_trues, y_preds)
    print(matrix)
    report = classification_report(y_trues, y_preds)
    print(report)

    end = timer()
    print(f"Elapsed time: {(end - start):.03f} seconds\n")

    # store the results in csv file    
    write_data(result_path, filenames, y_trues, y_preds)
    
    """
    Confusion matrix: In binary classification: 
    - True negatives is C0,0
    - False negatives is C1,0 
    - True positives is C1,1 
    - False positives is C0,1
                Pred N      Pred P
    Actual N     C0,0        C0,1
    Actual P     C1,0        C1,1
    """

