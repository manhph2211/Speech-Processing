import torch 
import torch.nn as nn 
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore", UserWarning)


class Trainer:
    
    def __init__(self,feature_extractor,classifier,criterion1,criterion2,optimizer1,optimizer2,loss_ratio=0.1,clip_value=1):

      self.feature_extractor = feature_extractor
      self.classifier = classifier
      self.criterion1 = criterion1
      self.criterion2 = criterion2
      self.optimizer1 = optimizer1
      self.optimizer2 = optimizer2
      self.loss_ratio = loss_ratio
      self.clip_value = clip_value

    def train_epoch(self,train_loader):

      self.compute_xvect.train()
      self.classify.train()
      train_loss_epoch = 0
      train_acc_epoch = []
      for em1,em2,label in tqdm(train_loader):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        em1 = em1.to(device)
        xvec1 = self.compute_xvect(em1)
        em2 = em2.to(device)
        xvec2 = self.compute_xvect(em2)
        label = label.to(device)
        xvec = torch.hstack((xvec1,xvec2))
        out = self.classify(xvec.reshape((xvec.shape[0],-1)))
        loss1 = self.criterion1(xvec1,xvec2,label)
        loss2 = self.criterion2(out,label.squeeze(dim=1))
        loss = loss1 * 0.1 + loss2
        train_loss_epoch+=loss.item()
        loss.backward()
        nn.utils.clip_grad_value_(self.compute_xvect.parameters(), clip_value=self.clip_value)
        nn.utils.clip_grad_value_(self.classify.parameters(), clip_value=self.clip_value)
        self.optimizer1.step()
        self.optimizer2.step()
        _, predict = out.max(dim=1)
        train_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
      return sum(train_acc_epoch)/len(train_acc_epoch), train_loss_epoch


    def val_epoch(self,test_loader):

      self.compute_xvect.eval()
      self.classify.eval()  
      val_loss_epoch = 0
      val_acc_epoch = []
      with torch.no_grad():
        for em1,em2,label in tqdm(test_loader):
          em1 = em1.to(device)
          xvec1 = self.compute_xvect(em1)
          em2 = em2.to(device)
          xvec2 = self.compute_xvect(em2)
          label = label.to(device)
          xvec = torch.hstack((xvec1,xvec2))
          out = self.classify(xvec.reshape((xvec.shape[0],-1)))
          loss1 = self.criterion1(xvec1,xvec2,label)
          loss2 = self.criterion2(out,label.squeeze(dim=1))
          loss = loss1 * 0.1 + loss2 
          val_loss_epoch+=loss.item()
          _, predict = out.max(dim=1)
          val_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
        return sum(val_acc_epoch)/len(val_acc_epoch), val_loss_epoch


    def save_checkpoint(self):
      compute_xvect.load_state_dict(torch.load(ckpt))
      classify.load_state_dict(torch.load('classify.pth'))
