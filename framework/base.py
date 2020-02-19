import torch
from abc import abstractmethod
import utils.data_tool as data_tool
import utils.file_tool as file_tool
import optuna
import utils.log_tool as log_tool
import utils.visualization_tool as visualization_tool
import numpy as np
import utils.SimpleProgressBar as progress_bar
import time

import framework


class Loss(torch.nn.Module):
    def __init__(self, arg_dict, regular_part_list, regular_factor_list):
        super().__init__()
        self.arg_dict = arg_dict
        self.regular_flag = arg_dict['regular_flag']
        self.regular_part_list = regular_part_list
        self.regular_factor_list = regular_factor_list

    def forward(self, model_outputs, labels):
        model_outputs = model_outputs.reshape(labels.size())
        cross_loss = torch.nn.BCELoss()(model_outputs, labels)
        batch_size = self.arg_dict['batch_size']
        regular_items = []
        if self.regular_flag:
            weights_list = []
            for part, factor in zip(self.regular_part_list, self.regular_factor_list):
                parameters_temp = part.named_parameters()
                weights_list.clear()
                for name, p in parameters_temp:
                    # print(name)
                    if (name.startswith('w') or name.find('weight') != -1) and (name.find('bias') == -1):
                        weights_list.append(p.reshape(-1))
                weights = torch.cat(weights_list, dim=0)
                # print(len(weights))
                para_sum = torch.pow(weights, 2).sum()
                regular_items.append((factor*para_sum)/(2*batch_size))
        result = cross_loss
        for item in regular_items:
            result += item
        correct_count = 0
        for output_, label in zip(model_outputs, labels):
            if torch.abs(output_-label) < 0.5:
                correct_count += 1
        return result, correct_count


class Framework(torch.nn.Module):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = self.create_arg_dict()
        self.update_arg_dict(arg_dict)
        self.data_type = self.arg_dict['dtype']
        gpu_id = self.arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

    def update_arg_dict(self, arg_dict):
        for name in arg_dict:
            self.arg_dict[name] = arg_dict[name]


    @abstractmethod
    def create_arg_dict(self):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def create_models(self):

        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def deal_with_example_batch(self, example_ids):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def get_regular_parts(self):
        raise RuntimeError("have not implemented this abstract method")

    @abstractmethod
    def get_input_of_visualize_model(self, example_ids, example_dict):
        raise RuntimeError("have not implemented this abstract method")


class FrameworkManager:
    def __init__(self, arg_dict, trial=None):
        super().__init__()
        self.arg_dict = arg_dict

        gpu_id = self.arg_dict['ues_gpu']
        if gpu_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu_id)

        corpus = self.arg_dict['corpus']
        self.data_manager = data_tool.DataManager(corpus())

        self.framework = None
        self.entire_model_state_dict_file = None
        self.optimizer_state_dict_file = None

        self.create_models()
        if not arg_dict['repeat_train']:
            self.framework.load_state_dict(torch.load(self.entire_model_state_dict_file))
            self.optimizer.load_state_dict(torch.load(self.optimizer_state_dict_file))

        self.framework_logger_name = 'framework_logger'
        if trial is not None:
            self.trial = trial
            self.trial_step = 0
            self.framework_logger_name += str(trial.number)

        self.data_loader_dict = self.data_manager.get_loader_dict(k_fold=arg_dict['k_fold'],
                                                                  batch_size=arg_dict['batch_size'], force=True)
        if self.framework.arg_dict['model_path'] is not None:
            model_path = self.framework.arg_dict['model_path']
            self.entire_model_state_dict_file = file_tool.connect_path(model_path, 'checkpoint/entire_model.pt')
            self.optimizer_state_dict_file = file_tool.connect_path(model_path, 'checkpoint/optimizer.pt')

        self.logger = log_tool.get_logger(self.framework_logger_name,
                                          file_tool.connect_path(self.framework.arg_dict['model_path'], 'log.txt'))

    def create_models(self):
        self.framework = self.get_framework()

        self.framework.create_models()

        self.losser = Loss(self.arg_dict, *self.framework.get_regular_parts())
        gpu_id = self.arg_dict['ues_gpu']
        if gpu_id == -1:
            self.framework.cpu()
        else:
            self.framework.cuda(self.device)

        if self.arg_dict['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.framework.parameters(), lr=self.arg_dict['learn_rate'],
                                             momentum=self.arg_dict['sgd_momentum'])

        elif self.arg_dict['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.framework.parameters(), lr=self.arg_dict['learn_rate'])
        else:
            raise ValueError

    def get_framework(self):
        if self.arg_dict['framework_name'] == "LSSE":
            arg_dict = self.arg_dict.copy()
            arg_dict['max_sentence_length'] = self.data_manager.get_max_sent_len()
            arg_dict['dep_kind_count'] = self.data_manager.get_max_dep_type()
            return framework.LSSEFramework(arg_dict)

    def __train_epoch__(self, epoch, loader):
        loss_avg = 0
        train_accuracy_avg = 0
        train_example_count = 0
        for b, batch in enumerate(loader):
            example_ids = batch['example_id']
            labels = batch['label'].to(device=self.device, dtype=self.framework.data_type)
            self.optimizer.zero_grad()
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # start_time = time.time()
            model_output = self.framework(example_ids, loader.example_dict)
            loss, train_correct_count = self.losser(model_output, labels)
            # end_time = time.time()
            # print('time:{}'.format(end_time-start_time))
            # start_time = time.time()
            loss.backward()
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # loss = float(loss)
            self.optimizer.step()
            loss_avg += float(loss)
            train_accuracy_avg += train_correct_count
            train_example_count += len(labels)
            print('epoch:{}  batch:{}  arg_loss:{}'.format(epoch + 1, b + 1, loss))
            # end_time = time.time()
            # print('time:{}'.format(end_time-start_time))
            # global_progress_bar.update((b+1)*100/len(train_loader))
        # print()
        loss_avg = loss_avg / len(loader)
        train_accuracy_avg = train_accuracy_avg / train_example_count
        return train_accuracy_avg, loss_avg

    def __train_fold__(self, train_loader, valid_loader):
        return_state = ""
        epoch = 0
        max_accuracy = 0
        max_accuracy_e = 0
        train_accuracy_list = []
        loss_list = []
        valid_accuracy_list = []
        valid_f1_list = []
        trial_count_report = 0
        trial_report_list = []
        try:
            global global_progress_bar
            best_result = None
            for epoch in range(self.arg_dict['epoch']):
                train_accuracy_avg, loss_avg = self.__train_epoch__(epoch, train_loader)
                self.logger.info(
                    'epoch:{} train_accuracy:{}  arg_loss:{}'.format(epoch + 1, train_accuracy_avg, loss_avg))

                with torch.no_grad():
                    evaluation_result = self.evaluation_calculation(valid_loader)
                    valid_accuracy = evaluation_result['metric']['accuracy']
                    self.logger.info(evaluation_result['metric'])

                    if valid_accuracy > max_accuracy:
                        best_result = evaluation_result['metric']
                        max_accuracy = valid_accuracy
                        max_accuracy_e = epoch + 1
                        self.save_model()

                    train_accuracy_list.append(train_accuracy_avg)
                    loss_list.append(loss_avg)
                    valid_accuracy_list.append(valid_accuracy)
                    valid_f1_list.append(evaluation_result['metric']['F1'])

                return_state = "finished_max_epoch"
                # print(1.0 - train_accuracy_avg)
                self.trial.report(1.0 - valid_accuracy, self.trial_step)
                self.logger.info('trial_report:{} at step:{}'.format(1.0 - valid_accuracy, self.trial_step))
                trial_report_list.append((1.0 - valid_accuracy, self.trial_step))
                self.trial_step += 1
                trial_count_report += 1
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if train_accuracy_avg >= 0.998:
                    if self.trial.number < self.arg_dict['start_up_trials']:
                        self.logger.info('trial_number: {} trial padding:{}'.format(self.trial.number,
                                                                                    self.arg_dict['epoch'] - epoch - 1))
                        for i in range(self.arg_dict['epoch'] - epoch - 1):
                            self.trial.report(1.0 - valid_accuracy, self.trial_step)
                            self.logger.info('trial_report:{} at step:{}'.format(1.0 - valid_accuracy, self.trial_step))
                            trial_report_list.append((1.0 - valid_accuracy, self.trial_step))
                            self.trial_step += 1
                            trial_count_report += 1
                            if self.trial.should_prune():
                                raise optuna.exceptions.TrialPruned()
                    return_state = "over_fitting"
                    break

            self.logger.info('trial_report_count:{}'.format(trial_count_report))

            if best_result is not None:
                self.logger.info(
                    'max acc:{}  F1:{}  best epoch:{}'.format(max_accuracy, best_result['F1'], max_accuracy_e))

        except KeyboardInterrupt:
            return_state = 'KeyboardInterrupt'
            if best_result is not None:
                self.logger.info(
                    'max acc:{}  F1:{}  best epoch:{}'.format(max_accuracy, best_result['F1'], max_accuracy_e))
            else:
                print('have not finished one epoch')
        max_accuracy = float(max_accuracy)

        record_dict = {
            'train_acc': train_accuracy_list,
            'loss': loss_list,
            'valid_acc': valid_accuracy_list,
            'valid_F1': valid_f1_list,
            'trial_report_list': trial_report_list
        }
        return round(1 - max_accuracy, 4), epoch + 1, return_state, record_dict

    def train_model(self):
        self.logger.info('begin to train model')
        train_loader_tuple_list = self.data_loader_dict['train_loader_tuple_list']
        best_result = (1, 0, None)
        record_list = []
        for tuple_index, train_loader_tuple in enumerate(train_loader_tuple_list, 1):
            self.create_models()
            train_loader, valid_loader = train_loader_tuple
            self.logger.info('train_loader:{}  valid_loader:{}'.format(len(train_loader), len(valid_loader)))
            self.logger.info('begin train {}-th fold'.format(tuple_index))

            result = self.__train_fold__(train_loader=train_loader, valid_loader=valid_loader)

            self.trial_step = self.arg_dict['epoch'] * tuple_index

            if result[0] < best_result[0]:
                best_result = result[0:3]
            record_list.append(result[3])

        record_file = file_tool.connect_path(self.arg_dict['model_path'], 'record_list.pkl')
        file_tool.save_data_pickle(record_list, record_file)
        return best_result

    def evaluation_calculation(self, data_loader):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        result = {}
        metric_dict = {}
        example_ids_dict = {}
        example_ids_fn = []
        example_ids_fp = []
        for batch in data_loader:
            example_ids = batch['example_id']
            labels = batch['label']
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # start_time = time.time()
            model_result = self.framework(example_ids, data_loader.example_dict)
            if len(model_result) != len(labels):
                raise ValueError

            for i in range(len(model_result)):
                output_ = model_result[i]
                label = labels[i]

                if (label != 1) and (label != 0):
                    raise ValueError

                if (output_ < 0) or (output_ > 1):
                    raise ValueError

                if output_ < 0.5:
                    if label == 0:
                        TN += 1
                    else:
                        FN += 1
                        example_ids_fn.append(int(example_ids[i]))

                elif output_ > 0.5:
                    if label == 1:
                        TP += 1
                    else:
                        FP += 1
                        example_ids_fp.append(int(example_ids[i]))

                else:
                    if label == 0:
                        FP += 1
                        example_ids_fp.append(int(example_ids[i]))
                    else:
                        FN += 1
                        example_ids_fn.append(int(example_ids[i]))

        example_ids_dict['FP'] = example_ids_fp
        example_ids_dict['FN'] = example_ids_fn
        metric_dict['TP'] = TP
        metric_dict['TN'] = TN
        metric_dict['FP'] = FP
        metric_dict['FN'] = FN
        metric_dict['accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        metric_dict['error'] = 1 - (TP + TN) / (TP + TN + FP + FN)
        if TP == 0:
            metric_dict['recall'] = 0
            metric_dict['precision'] = 0
        else:
            metric_dict['recall'] = TP / (TP + FN)
            metric_dict['precision'] = TP / (TP + FP)
        if (metric_dict['recall'] + metric_dict['precision']) == 0:
            metric_dict['F1'] = 0
        else:
            metric_dict['F1'] = 2 * metric_dict['recall'] * metric_dict['precision'] / (
                    metric_dict['recall'] + metric_dict['precision'])
        result['metric'] = metric_dict
        result['error_example_ids_dict'] = example_ids_dict
        return result

    def train_final_model(self):
        self.create_models()
        self.logger.info('begin to train final model')
        train_loader = self.data_manager.train_loader(self.arg_dict['batch_size'])
        train_accuracy_list = []
        loss_list = []
        for epoch in range(self.arg_dict['epoch']):
            train_accuracy_avg, loss_avg = self.__train_epoch__(epoch, train_loader)
            self.logger.info(
                'epoch:{} train_accuracy:{}  arg_loss:{}'.format(epoch + 1, train_accuracy_avg, loss_avg))

            train_accuracy_list.append(train_accuracy_avg)
            loss_list.append(loss_avg)

        record_dict = {
            'train_acc': train_accuracy_list,
            'loss': loss_list,
        }
        self.save_model()
        self.save_model(cpu=True)
        return record_dict

    def test_model(self):
        def get_save_data(error_example_ids):
            save_data = []
            for e_id in error_example_ids:
                example = example_dict[e_id]
                sentence1 = example.sentence1
                sentence2 = example.sentence2
                save_data.append(str(sentence1.id, sentence2.id))
                save_data.append(sentence1.original)
                save_data.append(sentence2.original)
            return save_data

        test_loader = self.data_manager.test_loader(self.arg_dict['batch_size'])
        with torch.no_grad():
            evaluation_result = self.evaluation_calculation(test_loader)
            self.logger.info(evaluation_result['metric'])
            example_dict = test_loader.example_dict
            fn_error_example_ids = evaluation_result['error_example_ids_dict']['FN']
            fp_error_example_ids = evaluation_result['error_example_ids_dict']['FP']
            fn_sava_data = get_save_data(fn_error_example_ids)
            fp_sava_data = get_save_data(fp_error_example_ids)

            file_tool.save_list_data(fn_sava_data, file_tool.connect_path(self.arg_dict['model_path'], 'error_file/fn_error_sentence_pairs.txt'),
                                     'w')
            file_tool.save_list_data(fp_sava_data, file_tool.connect_path(self.arg_dict['model_path'], 'error_file/fp_error_sentence_pairs.txt'),
                                     'w')
            return evaluation_result['metric']
        pass

    def visualize_model(self):
        train_loader = self.data_manager.train_loader(self.arg_dict['batch_size'])
        batch = iter(train_loader).next()
        example_ids = batch['example_id']
        input_data = self.framework.get_input_of_visualize_model(example_ids, train_loader.example_dict)
        filename = visualization_tool.create_log_filename()
        visualization_tool.log_graph(filename=filename, nn_model=self.framework, input_data=input_data, )

    def save_model(self, cpu=False):
        if cpu:
            self.framework.cpu()
            torch.save(self.framework.state_dict(), file_tool.PathManager.change_filename_by_append
                       (self.entire_model_state_dict_file, 'cpu'))
            self.framework.to(self.device)
        else:
            torch.save(self.framework.state_dict(), self.entire_model_state_dict_file)
        torch.save(self.optimizer.state_dict(), self.optimizer_state_dict_file)