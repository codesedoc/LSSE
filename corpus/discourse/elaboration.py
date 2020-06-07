from corpus.discourse.process_source_data import divide_examples as divide_example_dicts
import corpus.base_corpus as base_corpus


class Elaboration(base_corpus.Corpus):
    data_path = 'corpus/discourse/data'

    def create_examples(self):
        def create_examples_by_dicts(examples):
            example_obj_dict = {}
            for e in examples:
                satellite_id = str(e['satellite_id'])
                nucleus_id = str(e['nucleus_id'])
                satellite = base_corpus.Sentence(id_=satellite_id, original_sentence=e['satellite'])
                nucleus = base_corpus.Sentence(id_=satellite_id, original_sentence=e['nucleus'])

                id_ = int(e['id'])
                label = str(e['label'])
                example_obj = base_corpus.Example(id_, sentence1=satellite, sentence2=nucleus, label=label)
                if id_ in example_obj_dict:
                    raise ValueError("example in corpus is repeated")
                example_obj_dict[id_] = example_obj
            example_obj_list = list(example_obj_dict.values())
            return example_obj_list, example_obj_dict

        train_dicts, dev_dict, test_dicts = divide_example_dicts()

        self.train_example_list, self.train_example_dict = create_examples_by_dicts(train_dicts)
        self.dev_example_list, self.dev_example_dict = create_examples_by_dicts(dev_dict)
        self.test_example_list, self.test_example_dict = create_examples_by_dicts(test_dicts)

        self.example_dict = {}
        self.example_dict.update(self.train_example_dict)
        self.example_dict.update(self.dev_example_dict)
        self.example_dict.update(self.test_example_dict)

        if len(self.example_dict) != len(self.train_example_dict) + len(self.dev_example_dict) + len(self.test_example_dict):
            raise ValueError

        pass

    def create_sentences(self):
        pass

    def parse_sentences(self):
        pass

    def sentence_dict_from_examples(self):
        sentence_dict = {}
        for e in self.train_example_dict.values():
            sentence_dict[e.sentence1.id] = e.sentence1
            sentence_dict[e.sentence2.id] = e.sentence2

        for e in self.test_example_dict.values():
            sentence_dict[e.sentence1.id] = e.sentence1
            sentence_dict[e.sentence2.id] = e.sentence2

        return sentence_dict


singleton = None
def get_elaboration_obj(force=False):
    global singleton
    if force or (singleton is None):
        singleton = Elaboration()
        pass
    return singleton

def test():
    elaboration = get_elaboration_obj()

