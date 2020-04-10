from glue import *
from utils.general_tool import singleton
from glue.utils import InputFeatures, InputFeaturesWithGCN, TensorDictDataset
import utils.general_tool as general_tool
import logging
import torch
import utils.file_tool as file_tool
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


@singleton
class GLUE:
    def __init__(self):
        self.tasks_num_labels = glue_tasks_num_labels

        self.processors = glue_processors

        self.output_modes = glue_output_modes


@singleton
class GLUEManager:
    def __init__(self, args):
        self.glue = GLUE()
        self.args = args
        self.processor = self.glue.processors[self.args.task_name]()
        self.compute_metrics = compute_metrics

    def convert_examples_to_features(self, examples, tokenizer):
        framework_name = self.args.framework_name
        max_length = self.args.max_encoder_seq_length
        processor = self.processor
        label_list = processor.get_labels()
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

            if framework_name in ['LSSE', 'LSyE']:
                input_feature = InputFeaturesWithGCN(
                    self.args,
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                    example=example, sep_index=sep_index, text_a_len=text_a_len, text_b_len=text_b_len,
                    word_piece_flags=word_piece_flags
                )
            else:
                input_feature = InputFeatures(
                    self.args,
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                    example=example,
                )
            features.append(
                input_feature
            )
        return features

    def _load_and_cache_features(self, tokenizer, evaluate, test, local_rank=-1):
        args = self.args
        task = self.args.task_name
        model_type = self.args.model_type
        model_name_or_path = self.args.model_name_or_path
        max_seq_length = self.args.max_seq_length
        max_sent_length = self.args.max_sentence_length
        overwrite_cache = self.args.overwrite_cache
        data_dir = self.args.data_dir
        processor = self.processor

        if evaluate and test:
            raise ValueError

        if evaluate:
            data_set_type = 'dev'
        elif test:
            data_set_type = 'test'
        else:
            data_set_type = 'train'

        # Load data features from cache or dataset file
        cached_features_file = file_tool.connect_path(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                args.framework_name,
                data_set_type,
                list(filter(None, model_name_or_path.split("/"))).pop(),
                str(max_seq_length),
                str(task),
            ),
        )
        if file_tool.check_file(cached_features_file) and not overwrite_cache:
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

            # if args.framework_name in args.framework_with_gcn:
            #     processor.add_root_to_text_of_example()

            features = self.convert_examples_to_features(examples, tokenizer)
            # if local_rank in [-1, 0]:
            #     logger.info("Saving features into cached file %s", cached_features_file)
            #     torch.save(features, cached_features_file)

        if local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        return features

    def get_dataset(self, task, tokenizer, dev=False, test=False,):
        framework_name = self.args.framework_name
        output_mode = self.glue.output_modes[task]
        features = self._load_and_cache_features(tokenizer, dev, test)

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

        if framework_name in self.args.framework_with_gcn:
            all_sep_index = torch.tensor([f.sep_index for f in features], dtype=torch.int)
            all_word_piece_flags = torch.tensor([f.word_piece_flags for f in features], dtype=torch.int)

            all_text_a_org_len = torch.tensor([f.text_a_org_len for f in features], dtype=torch.int)
            all_text_a_len = torch.tensor([f.text_a_len for f in features], dtype=torch.int)
            all_adj_matrix_a = torch.tensor([f.adj_matrix_a for f in features], dtype=torch.int)
            all_sent_a_id = torch.tensor([f.sent_a_id for f in features], dtype=torch.int)

            all_text_b_org_len = torch.tensor([f.text_b_org_len for f in features], dtype=torch.int)
            all_text_b_len = torch.tensor([f.text_b_len for f in features], dtype=torch.int)
            all_adj_matrix_b = torch.tensor([f.adj_matrix_b for f in features], dtype=torch.int)

            dataset_item.update({
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
            })

        dataset = TensorDictDataset(dataset_item)
        return dataset