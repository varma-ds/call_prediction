import collections
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer():

    def __init__(self, model, optimizer, criterion, marked_criterion,
                 config, logger, data_loader, device,
                 metric_fns=None, scheduler=None, val_data_loader=None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.marked_criterion = marked_criterion
        self.config = config
        self.logger = logger
        self.metric_fns = metric_fns
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.device = device

        self.train_metrics = collections.defaultdict(lambda: utils.AverageMeter())
        self.valid_metrics = collections.defaultdict(lambda: utils.AverageMeter())

    def train_epoch(self, epoch):
        self.model.train()

        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            outputs = self.model(input_ids=data['ids'].to(self.device),
                                 attention_mask=data['masks'].to(self.device),
                                 token_type_ids=data['token_type_ids'].to(self.device))

            bs = data['ids'].shape[0]
            y_true = torch.zeros((bs, self.config.num_labels))
            y_true[range(bs), data['targets']] = 1
            
            loss = self.criterion(outputs['logits'], y_true.to(self.device))
            loss += self.marked_criterion(outputs['marked'].view(-1), 
                                          data['marked'].to(self.device))
            loss.backward()

            self.train_metrics['loss'].update(loss.item())

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if i % self.config.log_steps == 0:
                self.logger.info(f'loss at step {i}, {loss.item()}')

            if self.metric_fns:
                for metric_fn in self.metric_fns:
                    self.train_metrics[metric_fn.__name__].update(metric_fn(outputs, data))

        if self.val_data_loader:
            self.evaluate_epoch(epoch)

        results = {}
        for key, avg_meter in self.train_metrics.items():
            results[key] = avg_meter.avg

        for key, avg_meter in self.valid_metrics.items():
            results[key] = avg_meter.avg

        return results

    def evaluate_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_data_loader):
                outputs = self.model(input_ids=data['ids'].to(self.device),
                                     attention_mask=data['masks'].to(self.device),
                                     token_type_ids=data['token_type_ids'].to(self.device))

                bs = data['ids'].shape[0]
                y_true = torch.zeros((bs, self.config.num_labels))
                y_true[range(bs), data['targets']] = 1
                
                loss = self.criterion(outputs['logits'], y_true.to(self.device))
                loss += self.marked_criterion(outputs['marked'].view(-1), 
                                              data['marked'].to(self.device))
                self.valid_metrics['val_loss'].update(loss.item())

                if self.metric_fns:
                    for metric_fn in self.metric_fns:
                        self.valid_metrics[f'val_{metric_fn.__name__}'].update(metric_fn(outputs, data))

    def train(self):

        for epoch in range(self.config.num_epochs):
            results = self.train_epoch(epoch)

            logs = {'epoch': epoch}
            logs.update(results)

            for key, value in logs.items():
                self.logger.info(f'{key}: {value}')


class CallTrainer():

    def __init__(self, model, optimizer, criterion, marked_criterion, teacher_training,
                 config, logger, data_loader, device,
                 metric_fns=None, scheduler=None, val_data_loader=None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.marked_criterion = marked_criterion
        self.teacher_training = teacher_training
        self.config = config
        self.logger = logger
        self.metric_fns = metric_fns
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.device = device

        self.train_metrics = collections.defaultdict(lambda: utils.AverageMeter())
        self.valid_metrics = collections.defaultdict(lambda: utils.AverageMeter())

    def train_epoch(self, epoch):
        self.model.train()

        for i, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            outputs = self.model(data)

            loss = self.config.hibert_model_config['alpha1'] * \
                self.criterion(
                        outputs['finalreason_code'].reshape(-1, self.config.num_labels), 
                        data['targets'].reshape(-1, self.config.num_labels).to(self.device))

            loss += self.config.hibert_model_config['alpha2'] * \
                self.teacher_training(
                    F.log_softmax(outputs['reason_code'].reshape(-1, self.config.num_labels), dim=-1),
                    data['utt_bar'].reshape(-1, self.config.num_labels).to(self.device))

            loss += self.config.hibert_model_config['alpha3'] * \
                self.marked_criterion(outputs['marked'].squeeze(-1), 
                                      data['marked'].to(self.device))

            loss.backward()

            self.train_metrics['loss'].update(loss.item())
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if i % self.config.log_steps == 0:
                self.logger.info(f'loss at step {i}, {loss.item()}')

            if self.metric_fns:
                for metric_fn in self.metric_fns:
                    self.train_metrics[metric_fn.__name__].update(metric_fn(outputs, data))

        if self.val_data_loader:
            self.evaluate_epoch(epoch)

        results = {}
        for key, avg_meter in self.train_metrics.items():
            results[key] = avg_meter.avg

        for key, avg_meter in self.valid_metrics.items():
            results[key] = avg_meter.avg

        return results

    def evaluate_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_data_loader):
                outputs = self.model(data)

                loss = self.config.hibert_model_config['alpha1'] * \
                    self.criterion(
                            outputs['finalreason_code'].reshape(-1, self.config.num_labels), 
                            data['targets'].reshape(-1, self.config.num_labels).to(self.device))

                loss += self.config.hibert_model_config['alpha2'] * \
                    self.teacher_training(
                        F.log_softmax(outputs['reason_code'].reshape(-1, self.config.num_labels), dim=-1),
                        data['utt_bar'].reshape(-1, self.config.num_labels).to(self.device))

                loss += self.config.hibert_model_config['alpha3'] * \
                    self.marked_criterion(outputs['marked'].squeeze(-1), 
                                          data['marked'].to(self.device))

                self.valid_metrics['val_loss'].update(loss.item())

                if self.metric_fns:
                    for metric_fn in self.metric_fns:
                        self.valid_metrics[f'val_{metric_fn.__name__}'].update(metric_fn(outputs, data))

    def train(self):

        for epoch in range(self.config.num_epochs):
            results = self.train_epoch(epoch)

            logs = {'epoch': epoch}
            logs.update(results)

            for key, value in logs.items():
                self.logger.info(f'{key}: {value}')

