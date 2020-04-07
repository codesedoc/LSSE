# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os

from glue.utils import DataProcessor, InputExample, InputFeatures, InputFeaturesWithGCN, TensorDictDataset
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from glue.mrpc import MrpcProcessor
import utils.general_tool as general_tool
from transformers import glue_compute_metrics as compute_metrics


logger = logging.getLogger(__name__)


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_processors = {
    # "cola": ColaProcessor,
    # "mnli": MnliProcessor,
    # "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    # "sst-2": Sst2Processor,
    # "sts-b": StsbProcessor,
    # "qqp": QqpProcessor,
    # "qnli": QnliProcessor,
    # "rte": RteProcessor,
    # "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}


def glue_convert_examples_to_features(
    args,
    examples,
    tokenizer,
    framework_name,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
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

        sep_indexes = []
        sep_token = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
        for sep_index, id_ in enumerate(input_ids.copy()):
            if id_ == sep_token:
                sep_indexes.append(sep_index)

        if len(sep_indexes) != 2:
            raise ValueError

        sep_index = sep_indexes[0]
        text_a_len = sep_indexes[0] - 1
        text_b_len = sep_indexes[1] - sep_indexes[0] - 1

        word_piece_flags = general_tool.word_piece_flag_list(tokenizer.convert_ids_to_tokens(input_ids.copy()), '##')

        if text_a_len <= 0 or text_b_len <= 0:
            raise ValueError

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeaturesWithGCN(
                args,
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                example=example, sep_index=sep_index, text_a_len=text_a_len, text_b_len=text_b_len,
                word_piece_flags=word_piece_flags
            )
        )

    return features


# class MnliProcessor(DataProcessor):
#     """Processor for the MultiNLI data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["premise"].numpy().decode("utf-8"),
#             tensor_dict["hypothesis"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
#
#     def get_labels(self):
#         """See base class."""
#         return ["contradiction", "entailment", "neutral"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[8]
#             text_b = line[9]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
#
#
# class MnliMismatchedProcessor(MnliProcessor):
#     """Processor for the MultiNLI Mismatched data set (GLUE version)."""
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")
#
#
# class ColaProcessor(DataProcessor):
#     """Processor for the CoLA data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence"].numpy().decode("utf-8"),
#             None,
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             text_a = line[3]
#             label = line[1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#         return examples
#
#
# class Sst2Processor(DataProcessor):
#     """Processor for the SST-2 data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence"].numpy().decode("utf-8"),
#             None,
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = line[0]
#             label = line[1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#         return examples
#
#
# class StsbProcessor(DataProcessor):
#     """Processor for the STS-B data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence1"].numpy().decode("utf-8"),
#             tensor_dict["sentence2"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return [None]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[7]
#             text_b = line[8]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
#
#
# class QqpProcessor(DataProcessor):
#     """Processor for the QQP data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["question1"].numpy().decode("utf-8"),
#             tensor_dict["question2"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             try:
#                 text_a = line[3]
#                 text_b = line[4]
#                 label = line[5]
#             except IndexError:
#                 continue
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
#
#
# class QnliProcessor(DataProcessor):
#     """Processor for the QNLI data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["question"].numpy().decode("utf-8"),
#             tensor_dict["sentence"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")
#
#     def get_labels(self):
#         """See base class."""
#         return ["entailment", "not_entailment"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
#
#
# class RteProcessor(DataProcessor):
#     """Processor for the RTE data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence1"].numpy().decode("utf-8"),
#             tensor_dict["sentence2"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["entailment", "not_entailment"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples
#
#
# class WnliProcessor(DataProcessor):
#     """Processor for the WNLI data set (GLUE version)."""
#
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence1"].numpy().decode("utf-8"),
#             tensor_dict["sentence2"].numpy().decode("utf-8"),
#             str(tensor_dict["label"].numpy()),
#         )
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["0", "1"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[-1]
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


def load_and_cache_examples(
        args,
        task,
        tokenizer,
        model_type,
        model_name_or_path,
        framework_name,
        evaluate=False,
        test=False,
        max_seq_length=512,
        max_sent_length=50,
        overwrite_cache=False,
        data_dir=None,
        local_rank=-1
):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = glue_processors[task]()
    if not data_dir:
        data_dir = processor.data_path
    output_mode = glue_output_modes[task]

    if evaluate and test:
        raise ValueError

    if evaluate:
        data_set_type = 'dev'
    elif test:
        data_set_type = 'test'
    else:
        data_set_type = 'train'

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            data_set_type,
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_examples(data_dir=data_dir, set_type=data_set_type)
        )
        features = glue_convert_examples_to_features(
            args,
            examples,
            tokenizer,
            framework_name,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        )
        if local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        all_labels = None
        raise ValueError

    all_sep_index = torch.tensor([f.sep_index for f in features], dtype=torch.int)
    all_word_piece_flags = torch.tensor([f.word_piece_flags for f in features], dtype=torch.int)
    all_text_a_org_len = torch.tensor([f.text_a_org_len for f in features], dtype=torch.int)
    all_text_a_len = torch.tensor([f.text_a_len for f in features], dtype=torch.int)
    all_adj_matrix_a = torch.tensor([f.adj_matrix_a for f in features], dtype=torch.int)
    all_sent_a_id = torch.tensor([f.sent_a_id for f in features], dtype=torch.int)

    all_text_b_org_len = torch.tensor([f.text_b_org_len for f in features], dtype=torch.int)
    all_text_b_len = torch.tensor([f.text_b_len for f in features], dtype=torch.int)
    all_adj_matrix_b = torch.tensor([f.adj_matrix_b for f in features], dtype=torch.int)

    dataset_item = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'token_type_ids': all_token_type_ids,
        'labels': all_labels,
        'sep_index': all_sep_index,
        'word_piece_flags': all_word_piece_flags,
        'sent1_org_len': all_text_a_org_len,
        'sent1_len': all_text_a_len,
        'adj_matrix1': all_adj_matrix_a,
        'sent1_id': all_sent_a_id,
        'sent2_org_len': all_text_b_org_len,
        'sent2_len': all_text_b_len,
        'adj_matrix2': all_adj_matrix_b,
    }

    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    dataset = TensorDictDataset(dataset_item)
    return dataset

