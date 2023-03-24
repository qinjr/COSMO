import sys
import os
from logging import getLogger
from time import time
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.utils import set_color

from utils.evaluations import TopKMetric, PointMetric
from utils.early_stopper import EarlyStopper
from utils.log import dict2str, get_tensorboard, get_local_time, ensure_dir, combo_dict
from loss import BPRLoss, RegLoss, SoftmaxLoss

class AbsIndTrainer(object):
    r"""Independent Trainer Class is used to manage the training and evaluation processes of recommender system models in independent training mode.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model, dataset_name):
        self.config = config
        self.model = model
        self.dataset_name = dataset_name

    def fit(self):
        r"""Train the model based on the train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method [next] should be implemented.')

class Trainer(AbsIndTrainer):
    r"""Independent trainer for the warmup training phase
    """
    def __init__(self, config, model, dataset_name):
        super().__init__(config, model, dataset_name)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.train_mode = config['train_mode']
        if self.model.batch_random_neg:
            self.train_mode = 'batch_neg'
        self.batch_neg_size = config['batch_neg_size']
        self.loss_type = config['loss_type']
        self.sn_loss_type = config['sn_loss_type']
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.l2_norm = config['l2_norm']
        self.max_epochs = config['max_epochs']
        # how much training epoch will we conduct one evaluation
        self.eval_step = min(config['eval_step'], self.max_epochs)
        self.clip_grad_norm = config['clip_grad_norm']

        self.eval_batch_size = config['eval_batch_size']
        self.device = torch.device(config['device'])
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}-{}.pth'.format(self.model.get_name().lower(), dataset_name, get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.continue_metric = config['continue_metric']
        self.best_eval_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer(self.model.parameters())

        # for eval
        self.eval_mode = config['eval_mode']
        self.atten_study = config['atten_study']
        self.list_len = config['list_len']
        self.topks = config['topks']
        if 'user_is_single_dict' in config:
            with open(self.config['user_is_single_dict'],'rb') as f:
                self.user_is_single_dict = pkl.load(f)
        self.gauc = self.config['gauc'] if 'gauc' in self.config else None 

       
        self.user_f_pos = config['user_f_pos']
        self.item_f_pos = config['item_f_pos']
        self.sn_f_pos = config['sn_f_pos']

        self.lamda1 = config['lamda1']
        self.lamda2 = config['lamda2']
        
        # user hist
        if config['have_hist']:
            with open(self.config['hist_test_dict'], 'rb') as f:
                self.hist_test_dict = pkl.load(f)



    def _build_optimizer(self, params):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        # if self.config['reg_weight'] and self.weight_decay and self.weight_decay * self.config['reg_weight'] > 0:
        #     self.logger.warning(
        #         'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
        #         'which may lead to double regularization.'
        #     )
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _get_loss_func(self, loss_type):
        if loss_type == 'll':
            return torch.nn.BCELoss()
        elif loss_type == 'bpr':
            print(1)
            return BPRLoss()
        elif loss_type == 'reg':
            return RegLoss()
        elif loss_type == 'softmax':
            return SoftmaxLoss()
    
    def _get_user_hist(self, hist_indexs, stage = 'test'):
        user_history = []
        hist_len = []
        session_history = []
        hist_indexs = hist_indexs.tolist()
        if stage == 'test':
            for hist_index in hist_indexs:
                session,behavior = self.hist_test_dict[hist_index]
                user_history.append(behavior)
                session_history.append(session)
            
            user_history = torch.tensor(user_history)    
            session_history = torch.tensor(session_history)
            hist_len = torch.sum(user_history[:,:,0] !=0,axis=1)
            session_len = torch.sum(session_history[:,:,0] !=0,axis=1)

            session_len = torch.where(session_len>0, session_len,session_len+1 )

        return session_history, session_len, user_history, hist_len
    
    def _get_ubr_user_hist(self, x_user, x_item, stage ='train'):
        uids = x_user[:,0].tolist()
        tags = x_item[:,self.config['tags_num']].tolist()
        user_history = []
        hist_len = []
        if stage =='train':
            for i in range(len(uids)):
                if(tags[i] in self.ubr_hist_dict_train[uids[i]]):
                    user_history.append(self.ubr_hist_dict_train[uids[i]][tags[i]])
                    hist_len.append(self.ubr_hist_len_dict_train[uids[i]][tags[i]])
                else:
                    user_history.append(np.zeros([40,10],dtype=int))
                    hist_len.append(0)
        elif stage =='test':
            for i in range(len(uids)):
                if(tags[i] in self.ubr_hist_dict_test[uids[i]]):
                    user_history.append(self.ubr_hist_dict_test[uids[i]][tags[i]])
                    hist_len.append(self.ubr_hist_len_dict_test[uids[i]][tags[i]])
                else:
                    user_history.append(np.zeros([40,10],dtype=int))
                    hist_len.append(0)
        return torch.tensor(user_history), torch.tensor(hist_len)

    def _get_es_hist(self,x_user,x_item,stage ='train'):
        uids = x_user[:,0].tolist()
        items = x_item.tolist()
        def func(x):
            return list(map(str,x))
        records = list(map(func,items))
        uids = list(map(str,uids))
        records =[','.join(records[i]) for i in range(len(records))]
        uids =[str(uids[i])for i in range(len(uids))]
        queries = zip(uids,records)
        if stage =='train':
            user_history,hist_len = self.es_reader_train.query(queries,20,11)
            
        if stage == 'test':
            user_history,hist_len = self.es_reader_test.query(queries,20,11)
        user_history=np.array(user_history)[:,:,1:11]
        return torch.tensor(user_history), torch.tensor(hist_len)
        
    def _batch_neg_sample(self, x_item):
        idx = torch.randint(x_item.shape[0], (x_item.shape[0],))
        neg_item = x_item[idx]
        return neg_item, torch.zeros((neg_item.shape[0],)).to(self.device)

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.
        Args:
            epoch (int): the current epoch id
        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_eval_result': self.best_eval_result,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.
        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_eval_result = checkpoint['best_eval_result']

        # load architecture params from checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        des = 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        
        des = '%.' + str(des) + 'f'
        train_loss_output += set_color('train loss', 'blue') + ': ' + des % loss
        return train_loss_output + ']'

    def _train_epoch(self, train_dl, test_dl, epoch_idx, show_progress=True):
        self.model.train()
        main_loss_func = self._get_loss_func(self.loss_type)
 
        total_loss = None
        iter_data = (
            tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else train_dl
        )
        for batch_idx, batch_data in enumerate(iter_data):
            if self.loss_type == 'll':
                self.optimizer.zero_grad()
                
                user_hist = None
                hist_len = None

                x_session = None
                session_len = None
                x, y,x_session,user_hist,hist_labels = batch_data
                x_user, x_item, x_sn = x[:,self.user_f_pos], x[:,self.item_f_pos], x[:,self.sn_f_pos]
            
                ubr_user_hist,ubr_hist_len = None,None

                if self.model.use_hist:
                    if self.config['have_es']:
                        # user_hist, hist_len = self._get_user_hist(x_user,'train')
                        ubr_user_hist, ubr_hist_len = self._get_es_hist(x_user,x_item,'train')
                        # user_hist = user_hist.to(self.device)
                        # hist_len = hist_len.to(self.device)
                        user_hist,hist_len=None,None
                        ubr_user_hist = ubr_user_hist.to(self.device)
                        ubr_hist_len = ubr_hist_len.to(self.device)
                    else:
                        hist_len = torch.sum(user_hist[:,:,0] !=0,axis=1)
                        session_len = torch.sum(x_session[:,:,0]!=0, axis=1)
                        session_len = torch.where(session_len>0, session_len,session_len+1 )

                        user_hist = user_hist.to(self.device)
                        hist_len = hist_len.to(self.device)
                        x_session = x_session.to(self.device)
                        session_len = session_len.to(self.device)
                        
                        mask_hist = (user_hist[:,:,-1] !=0).to(self.device)
                        
                        
                        
                x_user = x_user.to(self.device)
                x_item = x_item.to(self.device)
                x_sn = x_sn.to(self.device)

                y = y.float().to(self.device)
                hist_labels = hist_labels.bool().to(self.device)

                #x_item = x_item[:,1].unsqueeze(1)
                pred,score1= self.model(x_user, x_item,x_sn, user_hist, hist_len,x_session,session_len,ubr_user_hist,ubr_hist_len)
                

                if self.train_mode == 'refine' and score1 is not None:
                    
                    if self.sn_loss_type == 'BPR':
                    
                        score1_pos = torch.mul(score1,hist_labels)
                        hist_labels_neg = torch.mul(~hist_labels,mask_hist)
                        score1_neg = torch.mul(score1,hist_labels_neg)
                        # score2_pos = torch.mul(score2,hist_labels)
                        # score2_neg = torch.mul(score2,torch.mul(~hist_labels,mask_hist))
                        

                        score1_pos = torch.sum(score1_pos,dim=1)/(torch.sum(hist_labels,dim=1)+1e-10) 
                        score1_neg = torch.sum(score1_neg,dim=1)/(torch.sum(hist_labels_neg,dim=1)+1e-10)
                        
                        bprloss = BPRLoss()

                        main_loss = main_loss_func(pred,y)
                        loss = self.lamda1 *main_loss + self.lamda2* bprloss(score1_pos,score1_neg)
                        # loss = self.lamda1 * main_loss_func(pred,y) + self.lamda2* bprloss(score1_pos,score1_neg) +self.lamda3* bprloss(score2_pos,score2_neg)
                    elif self.sn_loss_type == 'BCE':

                        hist_labels = hist_labels.float()
                        score1 = torch.nn.Sigmoid()(score1)
                        # score2 = torch.nn.Sigmoid()(score2)
                        score1 = torch.mul(score1,mask_hist)
                        # score2 = torch.mul(score2,mask_hist)

                        main_loss = main_loss_func(pred, y)
                        loss = self.lamda1* main_loss + self.lamda2* torch.nn.BCELoss()(score1, hist_labels) 
                else:
                    main_loss = main_loss_func(pred,y)
                    loss = main_loss

            elif self.loss_type == 'bpr':
                self.optimizer.zero_grad()
                
            
                x_pos,x_neg = batch_data
                x_user_pos, x_item_pos, x_sn_pos = x_pos[:,self.user_f_pos], x_pos[:,self.item_f_pos], x_pos[:,self.sn_f_pos]
                x_user_neg, x_item_neg, x_sn_neg = x_neg[:,self.user_f_pos], x_neg[:,self.item_f_pos], x_neg[:,self.sn_f_pos]
                
                x_user_pos = x_user_pos.to(self.device)
                x_item_pos = x_item_pos.to(self.device)
                x_sn_pos = x_sn_pos.to(self.device)
                x_user_neg = x_user_neg.to(self.device)
                x_item_neg = x_item_neg.to(self.device)
                x_sn_neg = x_sn_neg.to(self.device)

                pred_pos,score_pos = self.model(x_user_pos, x_item_pos,x_sn_pos,None,None,None,None,None,None)
                pred_neg,score_neg = self.model(x_user_neg,x_item_neg,x_sn_neg,None,None,None,None,None,None)
                main_loss = main_loss_func(pred_pos,pred_neg)
                loss = main_loss

            total_loss = main_loss.item() if total_loss is None else total_loss + main_loss.item()
            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

            

        return total_loss / len(train_dl)
    
    @torch.no_grad()
    def evaluate(self, dataloader, load_best_model=True, model_file=None, show_progress=False):
        if not dataloader:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        preds = []
        labels = []
        self.preds_and_labels_for_uid = {}
        uids = []


        # colect data
        for batch_idx, batch_data in enumerate(iter_data):
            
            x,y,hist_indexs = batch_data
            x_user, x_item,x_sn= x[:,self.user_f_pos], x[:,self.item_f_pos],x[:,self.sn_f_pos]
            user_hist = None
            hist_len = None
            ubr_user_hist, ubr_hist_len = None,None
            x_session,session_len = None,None

            if self.model.use_hist:
                if self.config['have_es']:
                    # user_hist, hist_len = self._get_user_hist(x_user,'test')
                    ubr_user_hist, ubr_hist_len = self._get_es_hist(x_user,x_item,'test')
                    # user_hist = user_hist.to(self.device)
                    # hist_len = hist_len.to(self.device)
                    user_hist = None
                    hist_len = None
                    ubr_user_hist = ubr_user_hist.to(self.device)
                    ubr_hist_len = ubr_hist_len.to(self.device)
                else:
                    x_session,session_len, user_hist, hist_len = self._get_user_hist(hist_indexs,'test')
                    

                    x_session = x_session.to(self.device)
                    session_len = session_len.to(self.device)
                    user_hist = user_hist.to(self.device)
                    hist_len = hist_len.to(self.device)

                    
            x_user = x_user.to(self.device)
            x_item = x_item.to(self.device)
            x_sn = x_sn.to(self.device)
            
            y = y.float().to(self.device)

            pred,score1 = self.model(x_user, x_item,x_sn, user_hist, hist_len, x_session, session_len, ubr_user_hist, ubr_hist_len)

            preds.append(pred)
            labels.append(y)
            uids.append(x_user[:,0])

        preds = torch.cat(preds)
        labels = torch.cat(labels)
        uids = torch.cat(uids)

        preds = preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        uids = uids.cpu().detach().numpy()

        # divide user_group to cal GAUC
        if self.gauc:
            single_gauc = []
            multi_gauc = []
            
            for i in range(int(len(preds)/self.list_len)):
                uid = uids[self.list_len*i]
                
                if uid in self.preds_and_labels_for_uid:
                    self.preds_and_labels_for_uid[uid][0].append(preds[self.list_len*i:self.list_len*(i+1)])
                    self.preds_and_labels_for_uid[uid][1].append(labels[self.list_len*i:self.list_len*(i+1)])
                else:
                    a= [preds[self.list_len*i:self.list_len*(i+1)]]
                    b = [labels[self.list_len*i:self.list_len*(i+1)]]
                    c= self.user_is_single_dict[uid]
                    self.preds_and_labels_for_uid[uid] = [a,b,c]
            for uid,value in self.preds_and_labels_for_uid.items():
                self.preds_and_labels_for_uid[uid][0] = np.concatenate(self.preds_and_labels_for_uid[uid][0])
                self.preds_and_labels_for_uid[uid][1] = np.concatenate(self.preds_and_labels_for_uid[uid][1])
                metrics = PointMetric(self.preds_and_labels_for_uid[uid][1],self.preds_and_labels_for_uid[uid][0])
                auc = metrics.cal_AUC()
                if self.preds_and_labels_for_uid[uid][2] == True:
                    single_gauc.append(auc)
                else:
                    multi_gauc.append(auc)
            single_gauc = np.mean(single_gauc)
            multi_gauc = np.mean(multi_gauc)

            gauc_result = {'single_gauc':single_gauc,'multi_gauc':multi_gauc}
        

        # get metrics
        metrics_point = PointMetric(labels, preds)
        eval_result_point = metrics_point.get_metrics()

        if self.config['eval_mode'] in ['all', 'list']:
            metrics_topk = TopKMetric(self.topks, self.list_len, labels, preds)
            eval_result_topk = metrics_topk.get_metrics()
            res = combo_dict([eval_result_topk, eval_result_point]) if self.config['eval_mode'] == 'all' else eval_result_topk
        elif self.config['eval_mode'] == 'point':
            res = eval_result_point

        if self.gauc:
            res = combo_dict([res,gauc_result])

        return res, pred, y


    def fit(self, train_dl, test_dl=None, verbose=True, saved=True, show_progress=True,ckpt_file = None):

        if ckpt_file:
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(ckpt_file)
            self.logger.info(message_output)
        
        
        if saved and self.start_epoch >= self.max_epochs:
            self._save_checkpoint(-1)
        
        early_stopper = EarlyStopper(self.config)
        
        if self.atten_study:
            eval_result = self.eval_atten_study(test_dl)
        else:
            eval_result, _, _ = self.evaluate(test_dl, False, show_progress=True)
        eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)
        test_dl.refresh()
        self.logger.info(eval_output)
        

        eval_results = []
        for epoch_idx in range(self.start_epoch, self.max_epochs):
            if epoch_idx > self.start_epoch:
                train_dl.refresh()
                test_dl.refresh()
            # train
            training_start_time = time()
            result = self.eval_atten_study(train_dl)
            train_dl.refresh()
            train_loss = self._train_epoch(train_dl, test_dl, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self.tensorboard.add_scalar('Loss/Train', train_loss, epoch_idx)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start_time = time()
                eval_result, pred, y = self.evaluate(test_dl, False, show_progress=True)
                

                eval_end_time = time()
                eval_results.append(eval_result)
                
                continue_metric_value = eval_result[self.continue_metric]
                continue_metric_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color(self.continue_metric, 'blue') + ": %f]") % \
                                     (epoch_idx, eval_end_time - eval_start_time, continue_metric_value)
                eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)

                if verbose:
                    self.logger.info(continue_metric_output)
                    self.logger.info(eval_output)
                self.tensorboard.add_scalar('eval_{}'.format(self.continue_metric), continue_metric_value, epoch_idx)
                continue_flag, save_flag = early_stopper.check(continue_metric_value, epoch_idx)
                if epoch_idx == 0:
                    save_flag = True

                if save_flag and saved:
                    self._save_checkpoint(epoch_idx)
                    save_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(save_output)

                if not continue_flag:
                    break
        
        best_epoch = early_stopper.get_best_epoch_idx()
        best_eval_result = eval_results[best_epoch]
        eval_output = set_color('best eval result', 'blue') + ': \n' + dict2str(best_eval_result)
        self.logger.info(eval_output)

        return best_eval_result



    def eval_atten_study(self, dataloader,show_progress=True):
        if not dataloader:
            return

        self.model.eval()

        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Atten_study   ", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        scores_pos = []
        scores_neg = []

        for batch_idx, batch_data in enumerate(iter_data):
            
            user_hist = None
            hist_len = None

            x_session = None
            session_len = None
            
            x, y,x_session,user_hist,hist_labels = batch_data
            x_user, x_item, x_sn = x[:,self.user_f_pos], x[:,self.item_f_pos], x[:,self.sn_f_pos]
            
            ubr_user_hist,ubr_hist_len = None,None

            if self.model.use_hist:
                hist_len = torch.sum(user_hist[:,:,0] !=0,axis=1)
                session_len = torch.sum(x_session[:,:,0]!=0, axis=1)
                session_len = torch.where(session_len>0, session_len,session_len+1 )

                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)
                x_session = x_session.to(self.device)
                session_len = session_len.to(self.device)
                        
                mask_hist = (user_hist[:,:,-1] !=0).to(self.device)
                        
                        
                        
            x_user = x_user.to(self.device)
            x_item = x_item.to(self.device)
            x_sn = x_sn.to(self.device)

            y = y.float().to(self.device)
            hist_labels = hist_labels.bool().to(self.device)

            pred,score= self.model(x_user, x_item,x_sn, user_hist, hist_len,x_session,session_len,ubr_user_hist,ubr_hist_len)
            
            att_score = torch.nn.Softmax(dim=1)(score)
            score_pos = torch.mul(att_score,hist_labels)
            hist_labels_neg = torch.mul(~hist_labels,mask_hist)
            score_neg = torch.mul(att_score,hist_labels_neg)
            
            score_pos = torch.sum(score_pos,dim=1)/(1e-10+torch.sum(hist_labels,dim=1))
            score_neg = torch.sum(score_neg,dim=1)/(1e-10+torch.sum(hist_labels_neg,dim=1))
            score_pos = score_pos.cpu().detach().numpy()
            score_neg = score_neg.cpu().detach().numpy()
            scores_pos.append(score_pos)
            scores_neg.append(score_neg)

            
        scores_pos = np.concatenate(scores_pos)
        scores_neg = np.concatenate(scores_neg)


        att_score_diff = np.mean(scores_pos - scores_neg)
        print('att_score_diff:{}'.format(att_score_diff))
        return {'att_score_diff':att_score_diff}
