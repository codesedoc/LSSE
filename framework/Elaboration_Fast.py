import torch
import numpy as np
import framework as fr
import time
import utils.file_tool as file_tool
import utils.parser_tool as parser_tool
import utils.general_tool as general_tool

import utils.data_tool as data_tool
from model import *
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
import corpus.discourse.elaboration as  elaboration
import copy
import transformers
from sklearn.metrics import matthews_corrcoef, f1_score


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dict):
        tensors = list(tensor_dict.values())
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        result = {}
        for key, tensor in self.tensor_dict.items():
            result[key] = tensor[index]
        return result

    def __len__(self):
        tensors = list(self.tensor_dict.values())
        return tensors[0].size(0)

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, args, **kwargs):
        self.args = args
        self.example = kwargs['example']
        self.e_id = self.example.id
        self.input_ids = kwargs['input_ids']
        self.attention_mask = kwargs['attention_mask']
        self.token_type_ids = kwargs['token_type_ids']
        self.label = kwargs['label']




    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Elaboration_Fast(torch.nn.Module):
    name = "Elaboration"
    result_path = file_tool.connect_path('result', name)

    def __init__(self, args):
        super().__init__()
        args.num_labels = 2
        args.output_mode = 'classification'
        self.args = args
        self.name = Elaboration_Fast.name
        self.result_path = Elaboration_Fast.result_path
        self.corpus = elaboration.get_elaboration_obj()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.bert_for_seq_classification = BertForSeqClassify(self.args)
        self.update_args()
        self.logger = log_tool.get_logger(self.name,
                                      file_tool.connect_path(self.args.output_dir, 'log.txt'))
        self.cuda(device=self.args.device)
        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None
        self.train_dataset = self.get_dataset('train')
        self.dev_dataset = self.get_dataset('dev')
        self.test_dataset = self.get_dataset('test')

    @classmethod
    def framework_name(cls):
        return cls.name

    def update_args(self):
        if self.args.output_mode == 'regression':
            self.args.fully_scales[-1] = 1
        if self.args.do_train:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            output_dir = file_tool.connect_path(self.result_path,
                                                'train',
                                                'bs:{}-lr:{}'.format(self.args.per_gpu_train_batch_size,
                                                                     self.args.learning_rate),
                                                time_str)

            file_tool.makedir(output_dir)
            if not file_tool.check_dir(output_dir):
                raise RuntimeError
            self.args.output_dir = output_dir

    def count_of_parameter(self):
        model_list = [self, self.bert_for_seq_classification]
        name_list = [self, self.bert_for_seq_classification.name]

        result = self._count_of_parameter(model_list=model_list, name_list=name_list)
        return result

    def forward(self, **kwargs):
        if len(kwargs) == 0:
            raise ValueError

        input_ids_batch = kwargs['input_ids']
        token_type_ids_batch = kwargs['token_type_ids']
        attention_mask_batch = kwargs['attention_mask']
        labels = kwargs['labels']

        outputs = self.bert_for_seq_classification(input_ids_batch, token_type_ids_batch, attention_mask_batch, labels)

        return outputs

    def optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.weight"]
        result = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0
                },
            ]

        return result

    def _convert_examples_to_features(self, tokenizer, examples):
        framework_name = self.args.framework_name
        max_length = self.args.max_encoder_seq_length
        label_list = ["0", "1"]
        output_mode = self.args.output_mode
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if self.args.model_type in ["xlnet"] else 0
        pad_on_left = bool(self.args.model_type in ["xlnet"])  # pad on the left for xlnet
        mask_padding_with_zero = True
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                print("Writing example %d/%d" % (ex_index, len_examples))

            inputs = tokenizer.encode_plus(example.sentence1.original, example.sentence2.original, add_special_tokens=True, max_length=max_length,)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
                len(attention_mask), max_length
            )
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), max_length
            )

            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < 3:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.id))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                self.logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                self.logger.info("label: %s (id = %d)" % (example.label, label))

            input_feature = InputFeatures(
                self.args,
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                example=example,
            )
            features.append(
                input_feature
            )
        return features

    def get_dataset(self, data_type='train'):
        def examples_to_dataset(examples):
            features = self._convert_examples_to_features(self.tokenizer, examples)
            output_mode = self.args.output_mode
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if output_mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            else:
                raise ValueError

            dataset_item = {
                'e_id': torch.tensor([f.e_id for f in features], dtype=torch.int),
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'token_type_ids': all_token_type_ids,
                'labels': all_labels
            }
            dataset = TensorDictDataset(dataset_item)
            return dataset

        if data_type == "train":
            if self.train_dataset is None:
                self.train_dataset = examples_to_dataset(self.corpus.train_example_list)
            return self.train_dataset
        elif data_type == "dev":
            if self.dev_dataset is None:
                self.dev_dataset = examples_to_dataset(self.corpus.dev_example_list)
            return self.dev_dataset
        elif data_type == "test":
            if self.test_dataset is None:
                self.test_dataset = examples_to_dataset(self.corpus.test_example_list)
            return self.test_dataset
        else:
            raise ValueError

    def my_train(self, train_dataset):
        """ Train the model """
        model = self
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

        optimizer_grouped_parameters = self.optimizer_grouped_parameters()


        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.base_learning_rate, eps=self.args.adam_epsilon,
                              weight_decay=0.0)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.scheduler = scheduler
        # Check if saved optimizer or scheduler states exist
        if file_tool.check_file(file_tool.connect_path(self.args.model_name_or_path, "optimizer.pt")) and \
           file_tool.check_file(file_tool.connect_path(self.args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(file_tool.connect_path(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(file_tool.connect_path(self.args.model_name_or_path, "scheduler.pt")))


        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0

        steps_trained_in_current_epoch = 0

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

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1


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

        metrics_result = acc_and_f1(np_predicts, np_labels)
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
            data_type = "dev"
        elif test_flag:
            prefix += "_test"
            data_type = "test"
        else:
            prefix += "_train"
            data_type = "train"
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        model = self
        eval_task_names = ("mnli", "mnli-mm") if self.args.task_name == "mnli" else (self.args.task_name,)
        eval_outputs_dirs = (self.args.output_dir, self.args.output_dir + "-MM") if self.args.task_name == "mnli" else (
                             self.args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = self.get_dataset(data_type)

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
            result = acc_and_f1(preds, out_label_ids)
            result['loss'] = eval_loss
            results.update(result)

            output_eval_file = file_tool.connect_path(eval_output_dir, "eval_{}_results.txt".format(prefix))
            with open(output_eval_file, "w") as writer:
                self.logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results

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

    def run(self):
        self.args.tensorboard_logdir = file_tool.connect_path('tensorboard',
                                                              str(socket.gethostname()),
                                                              self.args.framework_name,
                                                              self.get_name_in_result_path())
        self._print_args()
        # Training
        if self.args.do_train:
            train_dataset = self.get_dataset('train')
            global_step, tr_loss = self.my_train(train_dataset)
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
            # if not self.args.do_train:
            #     self.load(checkpoint)
            result = self.evaluate(dev_flag=True, test_flag=False, prefix=prefix)

        if self.args.do_test and self.args.local_rank in [-1, 0]:
            checkpoint = file_tool.connect_path(self.args.output_dir, 'checkpoint')
            self.logger.info("Evaluate the checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            # if not self.args.do_train:
            #     self.load(checkpoint)
            self.evaluate(dev_flag=False, test_flag=True, prefix=prefix)

        result['output_dir'] = self.args.output_dir
        optuna_result = 1-result['acc']
        return optuna_result, result

    def get_name_in_result_path(self):
        return 'blr-{}_lr-{}_bs-{}_ep-{}_td-{}_wd-{}_comfun-{}'.format(
            self.args.base_learning_rate,
            self.args.learning_rate,
            self.args.per_gpu_train_batch_size,
            self.args.num_train_epochs,
            self.args.transformer_dropout,
            self.args.weight_decay,
            self.args.semantic_compare_func,
        )

def run(args):
    general_tool.setup_seed(args.seed)
    model_ = Elaboration_Fast(args)
    model_.run()
