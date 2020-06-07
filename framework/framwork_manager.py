import torch
import utils.file_tool as file_tool
import utils.log_tool as log_tool
import framework as fr
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from glue.utils import AdamW, get_linear_schedule_with_warmup
import glue.glue_manager as glue_manager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from utils import general_tool
from model import MODEL_CLASSES
import socket

# import optuna
# import glob
# import utils.visualization_tool as visualization_tool
# import utils.data_tool as data_tool


class FrameworkManager:
    def __init__(self, args, trial=None):
        super().__init__()

        self.args = args
        self.glue_manager = glue_manager.GLUEManager(self.args)
        self.framework = None
        self.tokenizer = None
        self.framework_logger_name = 'framework_logger'
        self.logger = None
        if trial is not None:
            self.trial = trial
            self.trial_step = 0
            self.framework_logger_name += str(trial.number)

        # Prepare GLUE task
        self.args.task_name = self.args.task_name.lower()
        if self.args.task_name not in self.glue_manager.glue.processors:
            raise ValueError("Task not found: %s" % (self.args.task_name))
        processor = self.glue_manager.glue.processors[self.args.task_name]()
        self.args.output_mode = self.glue_manager.glue.output_modes[self.args.task_name]
        label_list = processor.get_labels()
        self.args.num_labels = len(label_list)
        self.args.dep_kind_count = processor.parse_info.dependency_count
        # Create framework
        self.create_framework()

    def create_framework(self):
        # Set seed
        general_tool.setup_seed(self.args.seed)

        # Load pretrained model and tokenizer
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.args.model_type = self.args.model_type.lower()
        _, _, self.tokenizer_class = MODEL_CLASSES[self.args.model_type]
        # self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', self.args.model_name_or_path,
        #                                 do_lower_case=self.args.do_lower_case,
        #                                 cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )

        self.framework = self.get_framework()

        self.framework.to(self.args.device)
        self.framework.name = self.args.framework_name
        file_tool.makedir(self.args.output_dir)
        self.logger = log_tool.get_logger(self.framework_logger_name,
                                          file_tool.connect_path(self.args.output_dir, 'log.txt'))

        self.logger.info('{} was created!'.format(self.framework.name))
        self._print_framework_parameter()

    def get_framework(self):
        frame_work = fr.frameworks[self.args.framework_name]
        return frame_work(self.args)

    def _print_framework_parameter(self):
        framework_parameter_count_dict = self.framework.count_of_parameter()
        self.logger.info('')
        self.logger.info("*" * 80)
        self.logger.info('{:^80}'.format("NN parameter count"))
        self.logger.info("*" * 80)

        self.logger.info('{:^20}{:^20}{:^20}{:^20}'.format('model name', 'total', 'weight', 'bias'))
        for item in framework_parameter_count_dict:
            self.logger.info('{:^20}{:^20}{:^20}{:^20}'.format(item['name'], item['total'], item['weight'], item['bias']))
        self.logger.info("*" * 80)

    def _print_args(self):
        self.logger.info('')
        self.logger.info("*" * 80)

        self.logger.info('{:^80}'.format("Training/evaluation arguments"))
        self.logger.info("*" * 80)

        arg_dict = vars(self.args).copy()
        # arg_dict['learning_rate'] = 0
        for key, value in arg_dict.items():
            self.logger.info('{}: {}'.format(key, value))
        self.logger.info("*" * 80)
        self.logger.info('')

    def train(self, train_dataset):
        """ Train the model """
        model = self.framework
        if self.args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(self.args.tensorboard_logdir)
            self.tb_writer = tb_writer


        train_sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) //
                                                                 self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)

        optimizer_grouped_parameters = self.framework.optimizer_grouped_parameters()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.args.base_learning_rate,
                                             momentum=self.args.sgd_momentum, weight_decay=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.base_learning_rate, eps=self.args.adam_epsilon,
                              weight_decay=0.0)
        else:
            raise ValueError
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Check if saved optimizer or scheduler states exist
        if file_tool.check_file(file_tool.connect_path(self.args.model_name_or_path, "optimizer.pt")) and \
           file_tool.check_file(file_tool.connect_path(self.args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(file_tool.connect_path(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(file_tool.connect_path(self.args.model_name_or_path, "scheduler.pt")))

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if file_tool.check_file(self.args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(self.args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.args.gradient_accumulation_steps)

            self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("  Continuing training from epoch %d", epochs_trained)
            self.logger.info("  Continuing training from global step %d", global_step)
            self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        # train_iterator = trange(
        #     epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        # )
        loss_list = []
        np_predicts = None
        np_labels = None
        general_tool.setup_seed(self.args.seed)  # Added here for reproductibility
        for epoch_index in range(int(self.args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                inputs = batch
                if self.args.model_type != "distilbert" and self.args.model_type not in ["bert", "xlnet", "albert"] :
                    del inputs["token_type_ids"]
                for key, t in inputs.items():
                    inputs[key] = t.to(self.args.device)
                # if self.args.model_type != "distilbert":
                #     inputs["token_type_ids"] = (
                #         batch[2] if self.args.model_type in ["bert", "xlnet", "albert"] else None
                #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                # loss_list.append(loss.item())
                logits = outputs[1]
                np_predicts, np_labels = self.append_np_predict_label(predicts=np_predicts,
                                             new_predicts=logits.detach().cpu().numpy().copy(),
                                             labels=np_labels,
                                             new_labels=inputs["labels"].detach().cpu().numpy().copy())

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    #     loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                    #     self.write_tensorboard_scalar(loss_scalar=loss_scalar, np_predicts=np_predicts, np_labels=np_labels, step=global_step)
                    #     np_predicts = None
                    #     np_labels = None
                    #     logging_loss = tr_loss

                    if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0 \
                            and not self.args.tune_hyper:
                        # Save model checkpoint
                        checkpoint_dir = file_tool.connect_path(self.args.output_dir, "checkpoint-{}".format(global_step))
                        if not file_tool.check_dir(checkpoint_dir):
                            file_tool.makedir(checkpoint_dir)
                        self.save(checkpoint_dir)
                        self.logger.info("Saving model checkpoint to %s", checkpoint_dir)
                        self.logger.info("Saving optimizer and scheduler states to %s", checkpoint_dir)

                    # tb_writer.add_scalar("loss", loss.item(), global_step)

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
            self.write_tensorboard_scalar(loss_scalar=loss_scalar, np_predicts=np_predicts, np_labels=np_labels,
                                          step=epoch_index)
            np_predicts = None
            np_labels = None
            logging_loss = tr_loss

        if self.args.local_rank in [-1, 0]:
            tb_writer.close()
        # file_tool.save_data_pickle(loss_list, 'analysis/baseline/run_on_my_pc/cuda/loss_list.pkl')
        return global_step, tr_loss / global_step

    def write_tensorboard_scalar(self, loss_scalar, np_predicts, np_labels, step):
        logs = {}
        if (
                self.args.local_rank == -1 and self.args.evaluate_during_training
        ):  # Only evaluate when single GPU otherwise metrics may not average well
            dev_results = self.evaluate(dev_flag=True, test_flag=False, prefix="step({})_during_training)".format(step))
            test_results = self.evaluate(dev_flag=False, test_flag=True, prefix="step({})_during_training".format(step))

            for dev_key, test_key in zip(dev_results, test_results):
                dev_log_key = "{}_dev".format(dev_key)
                logs[dev_log_key] = dev_results[dev_key]

                test_log_key = "{}_test".format(test_key)
                logs[test_log_key] = test_results[test_key]

            logs['dev_test_acc'] = (dev_results['acc'] + test_results['acc']) / 2

        learning_rate_scalar = self.scheduler.get_last_lr()[0]
        logs["learning_rate"] = learning_rate_scalar
        logs["loss"] = loss_scalar

        if self.args.output_mode == "classification":
            np_predicts = np.argmax(np_predicts, axis=1)
        elif self.args.output_mode == "regression":
            np_predicts = np.squeeze(np_predicts)

        metrics_result = self.glue_manager.compute_metrics(self.args.task_name, np_predicts, np_labels)
        logs['acc_train'] = metrics_result['acc']

        for key, value in logs.items():
            self.tb_writer.add_scalar(key, value, step)
        # print(json.dumps({**logs, **{"step": global_step}}))
        self.logger.info(json.dumps({**logs, **{"step": step}}))

    def append_np_predict_label(self, predicts, labels, new_predicts, new_labels):
        if predicts is None:
            predicts = new_predicts
            labels = new_labels
        else:
            predicts = np.append(predicts, new_predicts, axis=0)
            labels = np.append(labels, new_labels, axis=0)

        return predicts, labels

    def evaluate(self,  dev_flag, test_flag, prefix=""):
        if (dev_flag and test_flag) or not (dev_flag or test_flag):
            raise ValueError

        if dev_flag:
            prefix += "_dev"
        elif test_flag:
            prefix += "_test"
        else:
            prefix += "_train"
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        model = self.framework
        eval_task_names = ("mnli", "mnli-mm") if self.args.task_name == "mnli" else (self.args.task_name,)
        eval_outputs_dirs = (self.args.output_dir, self.args.output_dir + "-MM") if self.args.task_name == "mnli" else (
                             self.args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.glue_manager.get_dataset(task=eval_task, tokenizer=self.tokenizer, dev=dev_flag, test=test_flag)

            if not file_tool.check_dir(eval_output_dir) and self.args.local_rank in [-1, 0]:
                file_tool.makedir(eval_output_dir)

            self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

            # multi-gpu eval
            if self.args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            self.logger.info("***** Running evaluation {} *****".format(prefix))
            self.logger.info("  Num examples = %d", len(eval_dataset))
            self.logger.info("  Batch size = %d", self.args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            if not self.args.evaluate_during_training:
                bathes = tqdm(eval_dataloader, desc="Evaluating")
            else:
                bathes = eval_dataloader
            for batch in bathes:
                model.eval()
                inputs = batch
                with torch.no_grad():
                    if self.args.model_type != "distilbert" and self.args.model_type not in ["bert", "xlnet", "albert"]:
                        del inputs["token_type_ids"]
                    for key, t in inputs.items():
                        inputs[key] = t.to(self.args.device)

                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if self.args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif self.args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = self.glue_manager.compute_metrics(eval_task, preds, out_label_ids)
            result['loss'] = eval_loss
            results.update(result)

            output_eval_file = file_tool.connect_path(eval_output_dir, "eval_{}_results.txt".format(prefix))
            with open(output_eval_file, "w") as writer:
                self.logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results

    def run(self):
        self.args.tensorboard_logdir = file_tool.connect_path('tensorboard',
                                                              str(socket.gethostname()),
                                                              self.args.framework_name,
                                                              self.framework.get_name_in_result_path())
        self._print_args()
        # Training
        if self.args.do_train:
            train_dataset = self.glue_manager.get_dataset(task=self.args.task_name, tokenizer=self.tokenizer, dev=False, test=False)
            global_step, tr_loss = self.train(train_dataset)
            self.logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if self.args.do_train and (self.args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            checkpoint_dir = file_tool.connect_path(self.args.output_dir, 'checkpoint')
            if not file_tool.check_dir(checkpoint_dir) and self.args.local_rank in [-1, 0]:
                file_tool.makedir(checkpoint_dir)

            self.logger.info("Saving model checkpoint to %s", checkpoint_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            if self.args.task_name in ['QQP', 'qqp']:
                self.save(checkpoint_dir)

        # Evaluation
        result = None
        if self.args.do_eval and self.args.local_rank in [-1, 0]:
            checkpoint = file_tool.connect_path(self.args.output_dir, 'checkpoint')
            self.logger.info("Evaluate the checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if not self.args.do_train:
                self.load(checkpoint)
            result = self.evaluate(dev_flag=True, test_flag=False, prefix=prefix)

        if self.args.do_test and self.args.local_rank in [-1, 0]:
            checkpoint = file_tool.connect_path(self.args.output_dir, 'checkpoint')
            self.logger.info("Evaluate the checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if not self.args.do_train:
                self.load(checkpoint)
            self.evaluate(dev_flag=False, test_flag=True, prefix=prefix)

        result['output_dir'] = self.args.output_dir
        optuna_result = 1-result['acc']
        return optuna_result, result

    def load(self, checkpoint):
        self.framework.load_state_dict(torch.load(file_tool.connect_path(checkpoint, 'framework.pt')))
        self.tokenizer = self.tokenizer_class.from_pretrained(checkpoint)
        self.framework.to(self.args.device)

    def save(self, checkpoint):
        self.framework.cpu()
        torch.save(self.framework.state_dict(), file_tool.connect_path(checkpoint, 'framework.pt'))
        self.tokenizer.save_pretrained(checkpoint)
        torch.save(self.optimizer.state_dict(), file_tool.connect_path(checkpoint, 'optimizer.pt'))
        torch.save(self.scheduler.state_dict(), file_tool.connect_path(checkpoint, 'scheduler.pt'))
        self.framework.to(self.args.device)