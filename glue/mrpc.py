from glue.utils import DataProcessor, InputExample, InputSentence
import logging
import os
import utils.file_tool as file_tool
from utils.general_tool import singleton


logger = logging.getLogger(__name__)


@singleton
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def __init__(self):
        self.data_path = 'glue/data/MRPC'
        super().__init__()
        print('Create MrpcProcessor instance')

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        logger.info('Load {} set.'.format(set_type))
        e_id_interval = 10000
        e_id_base = {'train': 0, 'test': e_id_interval, 'dev': e_id_interval*2}[set_type]
        examples = []
        if len(lines)>e_id_interval:
            raise ValueError
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            e_id = e_id_base+i
            org_sent_a = line[3]
            org_sent_b = line[4]

            sent_a = self.org_sent2sent_obj_dict[org_sent_a]
            sent_b = self.org_sent2sent_obj_dict[org_sent_b]

            label = line[0]
            examples.append(InputExample(guid=guid, id=e_id, sent_a=sent_a, sent_b=sent_b, label=label))
        return examples

    def _create_sentence_dict(self, lines, set_type):
        sentence_set = set()
        sentence_dict = {}
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            sentence_set.add(line[3])
            sentence_set.add(line[4])

        for i, org_sent in enumerate(sentence_set):
            guid = "%s-%s" % (set_type, i)
            sent = InputSentence(guid=guid, original=org_sent)
            sentence_dict[guid] = sent

        return sentence_dict

    # def _create_parsed_file(self, sentence_dict):
    #     print('create parsed file')
    #     sent_id_dict = {}
    #     for sent_id, sent in sentence_dict.items():
    #         sent_id_dict[sent] = sent_id
    #     if len(sent_id_dict) != len(sentence_dict):
    #         raise ValueError
    #
    #     return examples