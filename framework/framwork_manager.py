import torch
import utils.data_tool as data_tool
import utils.file_tool as file_tool
import optuna
import utils.log_tool as log_tool
import utils.visualization_tool as visualization_tool
import framework as fr
import numpy as np
import utils.SimpleProgressBar as progress_bar
from sklearn.metrics import matthews_corrcoef, f1_score
import scipy
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from glue.utils import AdamW, get_linear_schedule_with_warmup
import glue.glue as glue
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from utils import general_tool
import glob
from model import MODEL_CLASSES
import os

class FrameworkManager:
    def __init__(self, args, trial=None):
        super().__init__()

        self.args = args

        self.framework = None
        self.framework_logger_name = 'framework_logger'
        self.logger = None
        if trial is not None:
            self.trial = trial
            self.trial_step = 0
            self.framework_logger_name += str(trial.number)
        self.create_framework()

    def create_framework(self):
        # Set seed
        general_tool.setup_seed(self.args.seed)

        # Prepare GLUE task
        self.args.task_name = self.args.task_name.lower()
        if self.args.task_name not in glue.glue_processors:
            raise ValueError("Task not found: %s" % (self.args.task_name))
        processor = glue.glue_processors[self.args.task_name]()
        self.args.output_mode = glue.glue_output_modes[self.args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.args.model_type = self.args.model_type.lower()

        # self.framework = self.get_framework()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        config = config_class.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=self.args.task_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.tokenizer_class = tokenizer_class
        self.tokenizer = tokenizer_class.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )

        self.framework = model_class.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )

        self.framework.to(self.args.device)
        self.framework.name = "SeE"
        file_tool.makedir(self.args.output_dir)
        self.logger = log_tool.get_logger(self.framework_logger_name,
                                          file_tool.connect_path(self.args.output_dir, 'log.txt'))

        self.logger.info('{} was created!'.format(self.framework.name))

    def get_framework(self):
        frame_work = fr.frameworks[self.args.framework_name]
        return frame_work(self.args)

    def train(self, train_dataset):
        """ Train the model """
        model = self.framework
        args = self.args
        logger = self.logger
        """ Train the model """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        # train_iterator = trange(
        #     epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        # )
        loss_list = []
        general_tool.setup_seed(args.seed)  # Added here for reproductibility
        for _ in range(int(args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                loss_list.append(loss.item())
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        if (
                                args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(args, model, self.tokenizer)
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        # model_to_save.save_pretrained(output_dir)
                        # tokenizer.save_pretrained(output_dir)

                        # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        self.save(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        file_tool.save_data_pickle(loss_list, 'loss_list_my.pkl')

        return global_step, tr_loss / global_step

    def evaluate(self, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        model = self.framework
        eval_task_names = ("mnli", "mnli-mm") if self.args.task_name == "mnli" else (self.args.task_name,)
        eval_outputs_dirs = (self.args.output_dir, self.args.output_dir + "-MM") if self.args.task_name == "mnli" else (
        self.args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = glue.load_and_cache_examples(
                task = eval_task,
                tokenizer = self.tokenizer,
                model_type = self.args.model_type,
                model_name_or_path = self.args.model_name_or_path,
                evaluate=True,
                max_seq_length=self.args.max_seq_length,
                max_sent_length=self.args.max_sentence_length,
                overwrite_cache=self.args.overwrite_cache,
                data_dir=self.args.data_dir
            )

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
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(self.args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if self.args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if self.args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
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
            result = glue.compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = file_tool.connect_path(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                self.logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results

    def run(self):
        self.logger.info("Training/evaluation parameters %s", self.args)
        # Training
        if self.args.do_train:
            train_dataset = glue.load_and_cache_examples(
                task=self.args.task_name,
                tokenizer=self.tokenizer,
                model_type=self.args.model_type,
                model_name_or_path=self.args.model_name_or_path,
                evaluate=False,
                max_seq_length=self.args.max_seq_length,
                max_sent_length=self.args.max_sentence_length,
                overwrite_cache=self.args.overwrite_cache,
                data_dir=self.args.data_dir
            )
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

            self.save(checkpoint_dir)

        # Evaluation
        result = None
        if self.args.do_eval and self.args.local_rank in [-1, 0]:
            checkpoint = file_tool.connect_path(self.args.output_dir, 'checkpoint')
            self.logger.info("Evaluate the checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            # self.load(checkpoint)
            result = self.evaluate(prefix=prefix)

        return result

    def load(self, checkpoint):
        self.framework.load_state_dict(torch.load(file_tool.connect_path(checkpoint, 'framework.pt')))
        self.tokenizer = self.tokenizer_class.from_pretrained(checkpoint)
        self.optimizer.load_state_dict(torch.load(file_tool.connect_path(checkpoint, 'optimizer.pt')))
        self.scheduler.load_state_dict(torch.load(file_tool.connect_path(checkpoint, 'scheduler.pt')))
        self.framework.to(self.args.device)

    def save(self, checkpoint):
        self.framework.cpu()
        torch.save(self.framework.state_dict(), file_tool.connect_path(checkpoint, 'framework.pt'))
        self.tokenizer.save_pretrained(checkpoint)
        torch.save(self.optimizer.state_dict(), file_tool.connect_path(checkpoint, 'optimizer.pt'))
        torch.save(self.scheduler.state_dict(), file_tool.connect_path(checkpoint, 'scheduler.pt'))
        self.framework.to(self.args.device)