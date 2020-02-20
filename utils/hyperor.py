import optuna
import utils.file_tool as file_tool
import math
import utils.general_tool as general_tool
import torch
import framework as fr


class Hyperor:
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.study_path = file_tool.connect_path("result/optuna", self.arg_dict['framework_name'])
        file_tool.makedir(self.study_path)
        self.study = optuna.create_study(study_name=self.arg_dict['framework_name'],
                                         storage='sqlite:///' + file_tool.connect_path(self.study_path, 'study_hyper_parameter.db'),
                                         load_if_exists=True,
                                         pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

        self.arg_dict['start_up_trials'] = self.study.pruner._n_startup_trials
        self.batch_size_list = [2, 4, 8, 16]
        self.learn_rate_list = [5e-5, 1e-5, 8e-6, 6e-6, 4e-6, 2e-6]
        self.trial_times = 30

    def objective(self, trial):
        # general_tool.setup_seed(1234)
        arg_dict = {
            'batch_size': self.batch_size_list[int(trial.suggest_discrete_uniform('batch_size_index', 0, len(self.batch_size_list)-1, 1))],
            'learn_rate': self.learn_rate_list[int(trial.suggest_discrete_uniform('learn_rate_factor', 0, len(self.learn_rate_list)-1, 1))],
            'gcn_layer': int(trial.suggest_discrete_uniform('gcn_hidden_layer', 2, 6, 2)),
        }

        self.arg_dict.update(arg_dict)

        framework_manager = fr.FrameworkManager(arg_dict=self.arg_dict, trial=trial)
        result, last_epoch, return_state = framework_manager.train_model()
        trial.set_user_attr('accuracy', 1 - result)
        trial.set_user_attr('last_epoch', last_epoch)
        trial.set_user_attr('return_state', return_state)
        torch.cuda.empty_cache()
        return result

    def tune_hyper_parameter(self):
        self.study.optimize(self.objective, n_trials=self.trial_times)
        self.study.set_user_attr('learn_rate_list', self.learn_rate_list)
        self.study.set_user_attr('batch_size_list', self.batch_size_list)
        file_tool.save_data_pickle(self.study, file_tool.connect_path(self.study_path, 'study_hyper_parameter.pkls'))
        # log_tool.model_result_logger.info(
        #     'Current best value is {} with parameters: {}.'.format(study.best_value, study.best_params))
