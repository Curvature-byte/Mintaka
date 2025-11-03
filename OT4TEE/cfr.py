# coding=utf-8
import argparse

import ot
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from data_processor import MyDataset
from models import *
from utils import *
from data_processor import *


def cal_wass(rep_0, rep_1, device, hparams):

    dist = ot.dist(rep_0, rep_1)

    gamma = ot.sinkhorn(
        torch.ones(len(rep_0), device=device) / len(rep_0),
        torch.ones(len(rep_1), device=device) / len(rep_1),
        dist.detach(),
        reg=hparams.get('epsilon'),
        stopThr=1e-4)

    loss_wass = torch.sum(gamma * dist)
    return loss_wass


class BaseEstimator:

    def __init__(self, board=None, hparams={}):
        data_name = hparams.get('data')
        print("Current data:", data_name)
        if data_name == "IHDP":
            train_set = MyDataset(f"Datasets/{data_name}/train_{hparams['seed']}.csv")
            traineval_set = MyDataset(f"Datasets/{data_name}/traineval_{hparams['seed']}.csv")
            eval_set = MyDataset(f"Datasets/{data_name}/eval_{hparams['seed']}.csv")
            test_set = MyDataset(f"Datasets/{data_name}/test_{hparams['seed']}.csv")
        else:
            train_set = MyDataset(f"Datasets/{data_name}/train.csv")
            traineval_set = MyDataset(f"Datasets/{data_name}/traineval.csv")
            eval_set = MyDataset(f"Datasets/{data_name}/eval.csv")
            test_set = MyDataset(f"Datasets/{data_name}/test.csv")
        self.device = torch.device(hparams.get('device'))
        if hparams['treat_weight'] == 0:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), drop_last=True)
        else:
            self.train_loader = DataLoader(train_set, batch_size=hparams.get('batchSize'), sampler=train_set.get_sampler(hparams['treat_weight']), drop_last=True)
        self.traineval_data = DataLoader(traineval_set, batch_size=256)  # for test in-sample metric
        self.eval_data = DataLoader(eval_set, batch_size=256)
        self.test_data = DataLoader(test_set, batch_size=256)

        # self.train_scaler = train_set.scaler
        # self.eval_scaler = eval_set.scaler
        # self.test_scaler = test_set.scaler

        self.train_metric = {
             "mae_ate": np.array([]),
             "mae_att": np.array([]),
             "pehe": np.array([]),
             "r2_f": np.array([]),
             "rmse_f": np.array([]),
             "r2_cf": np.array([]),
             "rmse_cf": np.array([]),
             "auuc": np.array([]),
             "rauuc": np.array([])}
        self.eval_metric = deepcopy(self.train_metric)
        self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {
             "mae_ate": None,
             "mae_att": None,
             "pehe": None,
             "r2_f": None,
             "rmse_f": None,
             "r2_cf": None,
             "rmse_cf": None,
             "auuc": None,
             "rauuc": None,}
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10 
        self.eval_best_metric["pehe"] = 100
        self.eval_best_metric['auuc'] = 0
        self.loss_metric = {'loss': np.array([]), 'loss_f': np.array([]), 'loss_c': np.array([])}

        self.epochs = hparams.get('epoch', 200)
        if hparams.get('model') == 'bnn-wass' or 'bnn-mmd':
            self.model = TLearner(train_set.x_dim, hparams).to(self.device)
        if hparams.get('model') == 'cfr-mmd' or 'cfr-wass':
            self.model = YLearner(train_set.x_dim, hparams).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.get('lr', 1e-3), weight_decay=hparams.get('l2_reg', 1e-4))
        self.hparams = hparams

        self.epoch = 0
        self.board = board

    def fit(self):

        iter_num = 0
        stop_epoch = 0 # record how many iterations the eval metrics do not improve
        for epoch in tqdm(range(1, self.epochs)):

            self.epoch = epoch
            self.model.train()
            for data in self.train_loader:  # train_loader
                self.model.zero_grad()
                data = data.to(self.device)
                _xt, _t, _yf, _ = data[:, :-2], data[:, -3], data[:, [-2]], data[:, -1]  
                _pred_f = self.model(_xt)

                # Section: loss calculation
                _loss_fit = self.criterion(_pred_f, _yf)
                _loss_wass = 0
                wass_indicator = (epoch > 20 and len(_t.unique()) > 1)
                if wass_indicator: # Avoid samples coming from same group

                    if self.hparams['model'] == 'cfr-wass' or self.hparams['model'] == 'bnn-wass':
                        _loss_wass = cal_wass(rep_0=self.model.rep_0,
                                              rep_1=self.model.rep_1,
                                              device=self.device,
                                              hparams=self.hparams)
                    elif self.hparams['model'] == 'cfr-mmd' or 'bnn-mmd':
                        _loss_wass = mmd2_rbf(self.model.rep_0, self.model.rep_1)

                _loss = _loss_fit + self.hparams['lambda'] * _loss_wass
                _loss.backward()
                self.optimizer.step()

                # Section: metric update
                _loss_wass = _loss_wass.item() if wass_indicator else 0
                self.board.add_scalar(
                    'loss/fit_loss',
                    _loss_fit.item(),
                    global_step=iter_num)
                self.board.add_scalar(
                    'loss/wass_loss',
                    _loss_wass,
                    global_step=iter_num)
                self.board.add_scalar(
                    'loss/total_loss',
                    _loss.item(),
                    global_step=iter_num)
                iter_num += 1

            # Section: evaluation and model selection
            if self.epoch % 10 == 0:
                _train_metric = self.evaluation(data='train')
                self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)
                [self.board.add_scalar(f"train/{key}", _train_metric[key], global_step=self.epoch) for key in _train_metric.keys()]

            if self.epoch % 1 == 0:
                _eval_metric = self.evaluation(data='eval')
                self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)
                [self.board.add_scalar(f"eval/{key}", _eval_metric[key], global_step=self.epoch) for key in _eval_metric.keys()]

                if abs(_eval_metric['auuc']) > abs(self.eval_best_metric['auuc']):
                # if abs(_eval_metric['pehe']) < abs(self.eval_best_metric['pehe']):
                    self.eval_best_metric = _eval_metric
                    self.train_best_metric = self.evaluation(data='train')
                    self.test_best_metric = self.evaluation(data='test')
                    stop_epoch = 0
                    print(self.eval_best_metric)
                else:
                    stop_epoch += 1
            if stop_epoch >= self.hparams['stop_epoch'] and self.epoch > 100:
                print(f'Early stop at epoch {self.epoch}')
                break

            self.epoch += 1

        # _test_metric = self.evaluation(data='test')
        # self.test_metric = metric_update(self.test_metric, _test_metric, self.epoch, )
        # [self.board.add_scalar(f"test/{key}", _test_metric[key], global_step=self.epoch) for key in _test_metric.keys()]

        [self.board.add_scalar(f"train_best/{key}", self.train_best_metric[key], global_step=self.epoch) for key in self.train_best_metric.keys()]
        [self.board.add_scalar(f"eval_best/{key}", self.eval_best_metric[key], global_step=self.epoch) for key in self.eval_best_metric.keys()]
        [self.board.add_scalar(f"test_best/{key}", self.test_best_metric[key], global_step=self.epoch) for key in self.test_best_metric.keys()]
        # self.eval_best_metric = {'eval_best/' + key: self.eval_best_metric[key] for key in self.eval_best_metric.keys()}
        # self.board.add_hparams(hparam_dict=self.hparams, metric_dict=self.eval_best_metric)
        self.board.add_graph(self.model, _xt)
        metric_export(path, self.train_best_metric, self.eval_best_metric, self.test_best_metric)

    def predict(self, dataloader):
        """

        :param dataloader
        :return: np.array, shape: (#sample)
        """
        self.model.eval()
        pred_0 = torch.tensor([], device=self.device)
        pred_1, yf, ycf, t = deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0),

        for data in dataloader:
            data = data.to(self.device)
            _x, _t, _yf, _ycf = data[:, :-3], data[:, [-3]], data[:, -2], data[:, -1]
            _x_0 = torch.cat([_x, torch.zeros_like(_t, device=self.device)], dim=-1)
            _x_1 = torch.cat([_x, torch.ones_like(_t, device=self.device)], dim=-1)
            _pred_0 = self.model(_x_0).reshape([-1])
            _pred_1 = self.model(_x_1).reshape([-1])
            pred_0 = torch.cat([pred_0, _pred_0], axis=-1)
            pred_1 = torch.cat([pred_1, _pred_1], axis=-1)
            yf = torch.cat([yf, _yf], axis=-1)
            ycf = torch.cat([ycf, _ycf], axis=-1)
            t = torch.cat([t, _t.reshape([-1])], axis=-1)

        pred_0 = pred_0.detach().cpu().numpy()
        pred_1 = pred_1.detach().cpu().numpy()
        yf = yf.cpu().numpy()
        ycf = ycf.cpu().numpy()
        t = t.detach().cpu().numpy()
        return pred_0, pred_1, yf, ycf, t

    def evaluation(self, data: str) -> dict():

        dataloader = {
            'train': self.traineval_data,
            'eval': self.eval_data,
            'test': self.test_data}[data]
        # scaler = {
        #     'train': self.train_scaler,
        #     'eval': self.eval_scaler,
        #     'test': self.test_scaler}[data]

        pred_0, pred_1, yf, ycf, t = self.predict(dataloader)
        # pred_0, pred_1, yf = scaler.reverse_y(pred_0), scaler.reverse_y(pred_1), scaler.reverse_y(yf)  # re-scale
        mode = 'in-sample' if data == 'train' else 'out-sample'
        metric = metrics(pred_0, pred_1, yf, ycf, t, mode)

        return metric


if __name__ == "__main__":

    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--model', type=str, default='cfr-mmd', help="['bnn-wass', 'cfr-mmd', 'cfr-wass']")
    hparams.add_argument('--device', type=str, default='cuda:1')
    hparams.add_argument('--data', type=str, default='ACIC')
    hparams.add_argument('--epoch', type=int, default=400)
    hparams.add_argument('--seed', type=int, default=1)
    hparams.add_argument('--stop_epoch', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')
    hparams.add_argument('--trial', type=str, default='MLResults')

    hparams.add_argument('--dim_backbone', type=str, default='60,60')
    hparams.add_argument('--dim_task', type=str, default='60,60')
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)
    hparams.add_argument('--treat_embed', type=bool, default=True)
    hparams.add_argument('--lambda', type=float, default=1.0, help='weight of wass_loss in loss function')
    hparams.add_argument('--ot_scale', type=float, default=0.1, help='weight of wass_loss in loss function')
    hparams.add_argument('--epsilon', type=float, default=1, help='weight of Entropy in sinkhorn')
    hparams = vars(hparams.parse_args())

    path = f"{hparams['trial']}/{hparams['data']}/{hparams['model']}_{hparams['batchSize']}_{hparams['seed']}"
    writer = SummaryWriter(path)
    torch.manual_seed(hparams['seed'])
    estimator = BaseEstimator(board=writer, hparams=hparams)
    estimator.fit()
    writer.close()
