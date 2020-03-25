import torch
import utils.data_tool as data_tool
import utils.file_tool as file_tool
import optuna
import utils.log_tool as log_tool
import utils.visualization_tool as visualization_tool
import framework as fr
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import utils.SimpleProgressBar as progress_bar
from sklearn.metrics import matthews_corrcoef, f1_score

# import time

# self.scheduler = get_linear_schedule_with_warmup(
#             self.optimizer, num_warmup_steps=self.arg_dict["warmup_steps"],
#             num_training_steps=int(len(train_loader) * self.arg_dict['epoch'])
#         )
from utils import general_tool


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

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

        self.data_loader_dict = self.data_manager.get_loader_dict(k_fold=arg_dict['k_fold'],
                                                                  batch_size=arg_dict['batch_size'], force=True)

        self.framework = None
        self.optimizer = None
        self.framework_logger_name = 'framework_logger'
        if trial is not None:
            self.trial = trial
            self.trial_step = 0
            self.framework_logger_name += str(trial.number)
        # self.create_framework()

    def __print_framework_parameter__(self):
        framework_parameter_count_dict = self.framework.count_of_parameter()
        self.logger.info("*" * 80)
        self.logger.info('{:^80}'.format("NN parameter count"))
        self.logger.info('{:^20}{:^20}{:^20}{:^20}'.format('model name', 'total', 'weight', 'bias'))
        for item in framework_parameter_count_dict:
            self.logger.info('{:^20}{:^20}{:^20}{:^20}'.format(item['name'], item['total'], item['weight'], item['bias']))
        self.logger.info("*" * 80)

    def create_framework(self):
        self.framework = self.get_framework()

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

        # self.loser = fr.BiCELo(self.arg_dict, *self.framework.get_regular_parts())

        checkpoint_path = file_tool.connect_path(self.framework.arg_dict['model_path'], 'checkpoint')
        file_tool.makedir(checkpoint_path)

        self.entire_model_state_dict_file = file_tool.connect_path(checkpoint_path, 'entire_model.pt')
        self.optimizer_state_dict_file = file_tool.connect_path(checkpoint_path, 'optimizer.pt')

        if not self.arg_dict['repeat_train']:
            if gpu_id == -1:
                self.framework.load_state_dict(torch.load(file_tool.PathManager.change_filename_by_append
                                                          (self.entire_model_state_dict_file, 'cpu')))
            else:
                self.framework.load_state_dict(torch.load(self.entire_model_state_dict_file))
            self.optimizer.load_state_dict(torch.load(self.optimizer_state_dict_file))

        self.logger = log_tool.get_logger(self.framework_logger_name,
                                          file_tool.connect_path(self.framework.arg_dict['model_path'], 'log.txt'))

        self.logger.info('{} was created!'.format(self.framework.name))

        self.__print_framework_parameter__()
        self.__print_framework_arg_dict__()

    def get_framework(self):
        arg_dict = self.arg_dict.copy()
        if 'max_sentence_length' not in arg_dict:
            arg_dict['max_sentence_length'] = self.data_manager.get_max_sent_len()

        arg_dict['dep_kind_count'] = self.data_manager.get_max_dep_type()
        frame_work = fr.frameworks[self.arg_dict['framework_name']]
        return frame_work(arg_dict)

    def __print_framework_arg_dict__(self):
        self.logger.info("*"*80)
        self.logger.info("framework args")
        for key, value in self.framework.arg_dict.items():
            self.logger.info('{}: {}'.format(key, value))
        self.logger.info("*" * 80)
        self.logger.info('\n')

    def __train_epoch__(self, epoch, loader):
        loss_avg = 0
        train_accuracy_avg = 0
        train_example_count = 0
        global_progress_bar = progress_bar.SimpleProgressBar()
        for b, batch in enumerate(loader):
            example_ids = batch['example_id']
            self.optimizer.zero_grad()
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # start_time = time.time()
            input_batch = self.framework.deal_with_example_batch(example_ids, loader.example_dict)
            loss, _ = self.framework(**input_batch)
            # end_time = time.time()
            # print('time:{}'.format(end_time-start_time))
            # start_time = time.time()
            loss.backward()
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # loss = float(loss)
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            loss_avg += float(loss.item())
            # train_example_count += len(labels)
            # print('epoch:{:5}  batch:{:5}  arg_loss:{:20}'.format(epoch + 1, b + 1, loss), end='\r')
            # end_time = time.time()
            # print('time:{}'.format(end_time-start_time))
            global_progress_bar.update((b+1)*100/len(loader))
        # print()
        loss_avg = loss_avg / len(loader)
        # train_accuracy_avg = train_accuracy_avg / train_example_count
        return loss_avg

    def __train_fold__(self, train_loader, valid_loader):
        return_state = ""
        epoch = 0
        max_accuracy = 0
        max_accuracy_e = 0
        loss_list = []
        valid_accuracy_list = []
        valid_f1_list = []
        trial_count_report = 0
        trial_report_list = []

        max_steps = self.arg_dict["max_steps"]
        train_epochs = self.arg_dict['epoch']
        if max_steps > 0:
            t_total = max_steps
            train_epochs = max_steps // len(train_loader) + 1
        else:
            t_total = len(train_loader) * train_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.arg_dict["warmup_steps"], num_training_steps=t_total
        )
        step_count = 0
        max_steps_break = False
        self.logger.info('total step:{} max step:{} warmup_steps:{} epoch:{} '.format(t_total, max_steps, self.arg_dict["warmup_steps"], train_epochs))
        try:
            best_result = None
            general_tool.setup_seed(self.arg_dict['seed'])
            for epoch in range(train_epochs):
                loss_avg = 0
                global_progress_bar = progress_bar.SimpleProgressBar()
                for b, batch in enumerate(train_loader):
                    example_ids = batch['example_id']
                    self.optimizer.zero_grad()

                    input_batch = self.framework.deal_with_example_batch(example_ids, train_loader.example_dict)
                    loss, _ = self.framework(**input_batch)

                    loss.backward()

                    self.optimizer.step()
                    if hasattr(self, 'scheduler'):
                        self.scheduler.step()
                        # self.logger.info("current learning rate:{}".format(self.scheduler.get_last_lr()[0]))
                    loss_avg += float(loss.item())
                    step_count += 1
                    if (max_steps > 0) and (step_count >= max_steps):
                        if max_steps_break:
                            raise RuntimeError
                        max_steps_break = True
                        break

                    global_progress_bar.update((b + 1) * 100 / len(train_loader))

                loss_avg = loss_avg / len(train_loader)

                self.logger.info(
                    'epoch:{}  arg_loss:{}'.format(epoch + 1, loss_avg))
                self.logger.info("current learning rate:{}".format(self.scheduler.get_last_lr()[0]))
                with torch.no_grad():
                    evaluation_result = self.evaluation_calculation(valid_loader)
                    valid_accuracy = evaluation_result['metric']['accuracy']
                    self.logger.info(evaluation_result['metric'])

                    if valid_accuracy > max_accuracy:
                        best_result = evaluation_result['metric']
                        max_accuracy = valid_accuracy
                        max_accuracy_e = epoch + 1
                        self.save_model()

                    loss_list.append(loss_avg)
                    valid_accuracy_list.append(valid_accuracy)
                    valid_f1_list.append(evaluation_result['metric']['F1'])

                return_state = "finished_max_epoch"
                # print(1.0 - train_accuracy_avg)
                if hasattr(self, 'trial'):
                    self.trial.report(1.0 - valid_accuracy, self.trial_step)
                    self.logger.info('trial_report:{} at step:{}'.format(1.0 - valid_accuracy, self.trial_step))
                    trial_report_list.append((1.0 - valid_accuracy, self.trial_step))
                    self.trial_step += 1
                    trial_count_report += 1
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if hasattr(self, 'trial'):
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

        record_dict = {
            'loss': loss_list,
            'valid_acc': valid_accuracy_list,
            'valid_F1': valid_f1_list,
            'trial_report_list': trial_report_list
        }
        return round(1 - max_accuracy, 4), epoch + 1, return_state, record_dict

    def train_model(self):
        train_loader_tuple_list = self.data_loader_dict['train_loader_tuple_list']
        avg_result = np.array([0, 0], dtype=np.float)
        record_list = []
        for tuple_index, train_loader_tuple in enumerate(train_loader_tuple_list, 1):
            # if tuple_index>2:
            #     break
            #repeat create framework, when each fold train
            self.create_framework()
            self.logger.info('{} was created!'.format(self.framework.name))
            train_loader, valid_loader = train_loader_tuple
            self.logger.info('train_loader:{}  valid_loader:{}'.format(len(train_loader), len(valid_loader)))
            self.logger.info('begin train {}-th fold'.format(tuple_index))
            result = self.__train_fold__(train_loader=train_loader, valid_loader=valid_loader)
            self.trial_step = self.arg_dict['epoch'] * tuple_index
            avg_result += np.array(result[0:2], dtype=np.float)
            record_list.append(result[3])

        record_file = file_tool.connect_path(self.framework.arg_dict['model_path'], 'record_list.pkl')
        file_tool.save_data_pickle(record_list, record_file)
        avg_result = (avg_result/len(train_loader_tuple_list)).tolist()
        avg_result.append('finish')
        self.logger.info('avg_acc:{}'.format(avg_result[0]))
        return avg_result

    def __classification_evaluation__(self, data_loader):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        result = {}
        metric_dict = {}
        example_ids_dict = {}
        example_ids_fn = []
        example_ids_fp = []
        all_predicts = []
        all_labels = []
        for batch in data_loader:
            example_ids = batch['example_id']
            # if example_ids[0, 0] == 3585:
            #     example_ids = example_ids
            labels = batch['label']
            # end_time = time.time()
            # print('time:{}'.format(end_time - start_time))
            # start_time = time.time()
            input_batch = self.framework.deal_with_example_batch(example_ids, data_loader.example_dict)
            _, predicts = self.framework(**input_batch)
            if len(predicts) != len(labels):
                raise ValueError
            all_labels.extend(labels.reshape(-1).tolist())
            all_predicts.extend(predicts.reshape(-1).tolist())
            for i in range(len(predicts)):
                pred = predicts[i]
                label = labels[i]

                if (label != 1) and (label != 0):
                    raise ValueError

                if (pred != 1) and (pred != 0):
                    raise ValueError

                if pred == 0:
                    if label == 0:
                        TN += 1
                    else:
                        FN += 1
                        example_ids_fn.append(int(example_ids[i]))
                else:
                    if label == 1:
                        TP += 1
                    else:
                        FP += 1
                        example_ids_fp.append(int(example_ids[i]))

        example_ids_dict['FP'] = example_ids_fp
        example_ids_dict['FN'] = example_ids_fn
        metric_dict['TP'] = TP
        metric_dict['TN'] = TN
        metric_dict['FP'] = FP
        metric_dict['FN'] = FN
        metric_dict['accuracy'] = float((TP + TN) / (TP + TN + FP + FN))
        metric_dict['error'] = 1 - float((TP + TN) / (TP + TN + FP + FN))
        if TP == 0:
            metric_dict['recall'] = 0
            metric_dict['precision'] = 0
        else:
            metric_dict['recall'] = float(TP / (TP + FN))
            metric_dict['precision'] = float(TP / (TP + FP))
        if (metric_dict['recall'] + metric_dict['precision']) == 0:
            metric_dict['F1'] = 0
        else:
            metric_dict['F1'] = 2 * metric_dict['recall'] * metric_dict['precision'] / (
                    metric_dict['recall'] + metric_dict['precision'])
        all_predicts = np.array(all_predicts, dtype=np.int)
        all_labels = np.array(all_labels, dtype=np.int)
        acc = (all_predicts == all_labels).mean()

        if round(metric_dict['accuracy'], 4) != round(acc, 4):
            raise ValueError("acc calculate error!")

        f1 = f1_score(y_true=all_labels, y_pred=all_predicts)

        if round(metric_dict['F1'], 4) != round(f1, 4):
            raise ValueError("acc calculate error!")

        metric_dict['F1'] = f1
        result['metric'] = metric_dict
        result['error_example_ids_dict'] = example_ids_dict
        return result

    def evaluation_calculation(self, data_loader):
        if self.arg_dict['task_type'] == 'classification':
            return self.__classification_evaluation__(data_loader)
        else:
            raise RuntimeError

    def train_final_model(self):
        # self.framework.print_arg_dict()
        self.create_framework()
        # return
        self.logger.info('begin to train final model')
        train_loader = self.data_manager.train_loader(self.arg_dict['batch_size'])
        # train_loader, _ = self.data_loader_dict['train_loader_tuple_list'][1]

        max_steps = self.arg_dict["max_steps"]
        train_epochs = self.arg_dict['epoch']
        if max_steps > 0:
            t_total = max_steps
            train_epochs = max_steps // len(train_loader) + 1
        else:
            t_total = len(train_loader) * train_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.arg_dict["warmup_steps"], num_training_steps=t_total
        )

        self.logger.info("train_loader:{}".format(len(train_loader)))
        loss_list = []

        max_steps_break = False
        step_count = 0
        general_tool.setup_seed(self.arg_dict['seed'])
        for epoch in range(train_epochs):
            loss_avg = 0
            global_progress_bar = progress_bar.SimpleProgressBar()
            for b, batch in enumerate(train_loader):
                example_ids = batch['example_id']
                self.optimizer.zero_grad()

                input_batch = self.framework.deal_with_example_batch(example_ids, train_loader.example_dict)
                loss, _ = self.framework(**input_batch)

                loss.backward()

                self.optimizer.step()
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()
                loss_avg += float(loss.item())
                step_count += 1
                if (max_steps > 0) and (step_count >= max_steps):
                    if max_steps_break:
                        raise RuntimeError
                    max_steps_break = True
                    break

                global_progress_bar.update((b + 1) * 100 / len(train_loader))

            loss_avg = loss_avg / len(train_loader)

            self.logger.info(
                'epoch:{}  arg_loss:{}'.format(epoch + 1, loss_avg))
            loss_list.append(loss_avg)
            self.logger.info("current learning rate:{}".format(self.scheduler.get_last_lr()[0]))

        record_dict = {
            'loss': loss_list,
        }

        self.save_model()
        self.save_model(cpu=True)
        return record_dict

    def test_model(self):
        def get_save_data(error_example_ids):
            save_data = []
            for e_id in error_example_ids:
                e_id = str(e_id)
                example = example_dict[e_id]
                sentence1 = example.sentence1
                sentence2 = example.sentence2
                save_data.append(str([sentence1.id, sentence2.id]))
                save_data.append(sentence1.original)
                save_data.append(sentence2.original)
            return save_data
        if not self.arg_dict['repeat_train']:
            self.create_framework()
        test_loader = self.data_manager.test_loader(self.arg_dict['batch_size'])
        self.logger.info('test_loader length:{}'.format(len(test_loader)))
        with torch.no_grad():
            evaluation_result = self.evaluation_calculation(test_loader)
            self.logger.info(evaluation_result['metric'])
            example_dict = test_loader.example_dict
            fn_error_example_ids = evaluation_result['error_example_ids_dict']['FN']
            fp_error_example_ids = evaluation_result['error_example_ids_dict']['FP']
            fn_sava_data = get_save_data(fn_error_example_ids)
            fp_sava_data = get_save_data(fp_error_example_ids)

            error_file_path = file_tool.connect_path(self.framework.arg_dict['model_path'], 'error_file')
            file_tool.makedir(error_file_path)

            file_tool.save_list_data(fn_sava_data,
                                     file_tool.connect_path(error_file_path, 'fn_error_sentence_pairs.txt'), 'w')
            file_tool.save_list_data(fp_sava_data,
                                     file_tool.connect_path(error_file_path, 'fp_error_sentence_pairs.txt'), 'w')
            return evaluation_result['metric']
        pass

    def visualize_model(self):
        self.create_framework()
        train_loader = self.data_manager.train_loader(self.arg_dict['batch_size'])
        batch = iter(train_loader).next()
        example_ids = batch['example_id']
        input_data = self.framework.get_input_of_visualize_model(example_ids, train_loader.example_dict)
        visualization_path = file_tool.connect_path(self.framework.result_path, 'visualization')
        file_tool.makedir(visualization_path)
        filename = visualization_tool.create_filename(visualization_path)
        visualization_tool.log_graph(filename=filename, nn_model=self.framework, input_data=input_data)

    def save_model(self, cpu=False):
        if cpu:
            self.framework.cpu()
            torch.save(self.framework.state_dict(), file_tool.PathManager.change_filename_by_append
                       (self.entire_model_state_dict_file, 'cpu'))
            self.framework.to(self.device)
        else:
            torch.save(self.framework.state_dict(), self.entire_model_state_dict_file)
        torch.save(self.optimizer.state_dict(), self.optimizer_state_dict_file)