import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc

import torch.optim
import torch.nn.parallel


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            self.writer = None
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None
        # 添加保存最佳tAUC的变量
        self.best_mean_auc = 0
        self.time_points = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        # 时间点会在第一次调用train和validate函数时确定

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            c_index, mean_auc = self.validate(val_loader, model, criterion)
            # 评估模式下只返回两个值
            return c_index, mean_auc

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index, mean_auc = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_mean_auc = mean_auc
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score,
                    'best_mean_auc': self.best_mean_auc})
            print(' *** best c-index={:.4f}, best mean tAUC={:.4f} at epoch {}'.format(
                self.best_score, self.best_mean_auc, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        # 只返回两个值：最佳c-index和最佳mean_auc
        return self.best_score, self.best_mean_auc

    def calculate_safe_time_points(self, event_times, censorships):
        """计算安全的时间点，确保在有效范围内"""
        # 找出非审查样本（事件已发生）的时间
        uncensored_times = event_times[censorships == 0]

        if len(uncensored_times) == 0:
            print("Warning: No uncensored events found, using all event times for tAUC time points")
            uncensored_times = event_times

        # 确保时间点在有效范围内
        # sksurv要求时间点在区间[min_time+epsilon, max_time-epsilon]内
        epsilon = 1.0  # 安全边界
        min_time = np.min(uncensored_times) + epsilon
        max_time = np.max(uncensored_times) - epsilon

        # 确保min_time < max_time
        if min_time >= max_time:
            # 如果区间太小，使用单一时间点
            print("Warning: Valid time range too small, using single time point for tAUC")
            return np.array([(min_time + max_time) / 2])

        # 生成4个均匀分布的时间点，确保在有效范围内
        time_points = np.linspace(min_time, max_time, num=4)
        print(f"Selected time points for tAUC evaluation: {[float(t) for t in time_points]}")
        return time_points

    def prepare_survival_data(self, event_times, censorships, risk_scores):
        """准备用于sksurv评估的数据格式"""
        # 创建structured array，符合sksurv需要的格式
        dtypes = [('event', bool), ('time', float)]
        y = np.zeros(len(event_times), dtype=dtypes)
        y['event'] = (1 - censorships).astype(bool)
        y['time'] = event_times
        return y, risk_scores

    def train(self, train_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0

        # 初始化数组来收集所有批次的数据
        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        dataloader = tqdm(train_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (
        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
        c) in enumerate(dataloader):

            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S, P, P_hat, G, G_hat = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                   x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                   x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            # 收集数据到列表中
            all_risk_scores.append(risk.item())
            all_censorships.append(c.item())
            all_event_times.append(event_time.item())

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 转换为numpy数组
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)

        # 计算安全的时间点（每次都重新计算确保在范围内）
        self.time_points = self.calculate_safe_time_points(all_event_times, all_censorships)

        # 计算tAUC
        y_train, scores = self.prepare_survival_data(all_event_times, all_censorships, all_risk_scores)

        # 计算每个时间点的AUC
        auc_scores = []
        for t in self.time_points:
            try:
                # 将numpy标量转换为Python标量
                t_value = float(t)
                auc, _ = cumulative_dynamic_auc(y_train, y_train, scores, t_value)
                auc_scores.append(float(auc))  # 确保是Python float
            except Exception as e:
                print(f"Warning: Error calculating tAUC at time {t}: {str(e)}")
                auc_scores.append(float('nan'))

        # 计算有效AUC的平均值
        valid_aucs = [auc for auc in auc_scores if not np.isnan(auc)]
        mean_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else float('nan')
        if not np.isnan(mean_auc):
            mean_auc = float(mean_auc)  # 确保是Python float

        # 计算loss和c-index
        train_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        print('loss: {:.4f}, c_index: {:.4f}, mean tAUC: {:.4f}'.format(train_loss, c_index, mean_auc))

        # 正确格式化输出时间点和AUC值
        time_points_str = '[' + ', '.join(['{:.2f}'.format(float(t)) for t in self.time_points]) + ']'
        auc_values_str = '[' + ', '.join(
            ['{:.4f}'.format(auc) if not np.isnan(auc) else 'N/A' for auc in auc_scores]) + ']'
        print(f'tAUC at time points {time_points_str}: {auc_values_str}')

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)
            if not np.isnan(mean_auc):
                self.writer.add_scalar('train/mean_tAUC', mean_auc, self.epoch)
                for i, (t, auc) in enumerate(zip(self.time_points, auc_scores)):
                    if not np.isnan(auc):
                        self.writer.add_scalar(f'train/tAUC_at_{float(t):.1f}', auc, self.epoch)

    def validate(self, val_loader, model, criterion):
        model.eval()
        val_loss = 0.0

        # 初始化数组来收集所有批次的数据
        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        dataloader = tqdm(val_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (
        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
        c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                hazards, S, P, P_hat, G, G_hat = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                       x_omic3=data_omic3,
                                                       x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)

            risk = -torch.sum(S, dim=1).cpu().numpy()

            # 收集数据到列表中
            all_risk_scores.append(risk.item())
            all_censorships.append(c.cpu().numpy().item())
            all_event_times.append(event_time.item())

            val_loss += loss.item()

        # 转换为numpy数组
        all_risk_scores = np.array(all_risk_scores)
        all_censorships = np.array(all_censorships)
        all_event_times = np.array(all_event_times)

        # 如果时间点未设置，先计算
        if self.time_points is None:
            self.time_points = self.calculate_safe_time_points(all_event_times, all_censorships)

        # 计算tAUC
        y_val, scores = self.prepare_survival_data(all_event_times, all_censorships, all_risk_scores)

        # 计算每个时间点的AUC
        auc_scores = []
        for t in self.time_points:
            try:
                # 将numpy标量转换为Python标量
                t_value = float(t)
                auc, _ = cumulative_dynamic_auc(y_val, y_val, scores, t_value)
                auc_scores.append(float(auc))  # 确保是Python float
            except Exception as e:
                print(f"Warning: Error calculating tAUC at time {t}: {str(e)}")
                auc_scores.append(float('nan'))

        # 计算有效AUC的平均值
        valid_aucs = [auc for auc in auc_scores if not np.isnan(auc)]
        mean_auc = np.mean(valid_aucs) if len(valid_aucs) > 0 else float('nan')
        if not np.isnan(mean_auc):
            mean_auc = float(mean_auc)  # 确保是Python float

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        print('loss: {:.4f}, c_index: {:.4f}, mean tAUC: {:.4f}'.format(val_loss, c_index, mean_auc))

        # 正确格式化输出时间点和AUC值
        time_points_str = '[' + ', '.join(['{:.2f}'.format(float(t)) for t in self.time_points]) + ']'
        auc_values_str = '[' + ', '.join(
            ['{:.4f}'.format(auc) if not np.isnan(auc) else 'N/A' for auc in auc_scores]) + ']'
        print(f'tAUC at time points {time_points_str}: {auc_values_str}')

        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
            if not np.isnan(mean_auc):
                self.writer.add_scalar('val/mean_tAUC', mean_auc, self.epoch)
                for i, (t, auc) in enumerate(zip(self.time_points, auc_scores)):
                    if not np.isnan(auc):
                        self.writer.add_scalar(f'val/tAUC_at_{float(t):.1f}', auc, self.epoch)

        return c_index, mean_auc

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{mean_auc:.4f}_{epoch}.pth.tar'.format(
                                              score=state['best_score'],
                                              mean_auc=state['best_mean_auc'],
                                              epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
