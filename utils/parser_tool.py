import utils.number_tool as number_tool
import numpy as np
import utils.file_tool as file_tool
import utils.SimpleProgressBar as progress_bar


dependency2group_dict = {
    # 'root': 'group0',

    'dep': 'group1',
    'aux': 'group1',
    'auxpass': 'group1',
    'cop': 'group1',

    'ccomp': 'group2',
    'xcomp': 'group2',
    'dobj': 'group2',
    'iobj': 'group2',

    'nsubj': 'group3',
    'nsubjpass': 'group3',
    'csubj': 'group3',
    'csubjpass': 'group3',

    'cc': 'group4',
    'conj': 'group4',
    'expl': 'group4',
    'amod': 'group4',
    'appos': 'group4',
    'advcl': 'group4',
    'advmod': 'group4',
    'det': 'group4',
    'mwe': 'group4',
    'mark': 'group4',
    'neg': 'group4',
    'parataxis': 'group4',
    'ref': 'group4',
    'nmod': 'group4',
    'nummod': 'group4',
    'compound': 'group4',
    'case': 'group4',
    'acl': 'group4',

    'discourse': 'group5',

    # 'list': 'group6',
}
group_count = len(set(dependency2group_dict.values()))


class ParseInfo:
    def __init__(self, **kwargs):
        super().__init__()

        self.numeral_sentence_dict = kwargs['numeral_sentence_dict']
        self.dependency_dict = kwargs['dependency_dict']
        self.dependency_count = len(self.dependency_dict)
        self.id2dependency = kwargs['id2dependency']
        self.sent_len_table = np.zeros(500, np.longlong)
        avg_sent_len = 0
        max_sent_len = 0
        max_sent_id = 0
        for sent_id, number_sentence in self.numeral_sentence_dict.items():
            tokens = number_sentence['words']
            self.sent_len_table[len(tokens)] += 1
            avg_sent_len += len(tokens)
            if len(tokens) > max_sent_len:
                max_sent_len = len(tokens)
                max_sent_id = sent_id
        if self.sent_len_table[0] !=0:
            raise ValueError("exist empty sentence")
        self.avg_sent_len = round(avg_sent_len/len(self.numeral_sentence_dict),2)
        self.max_sent_len = max_sent_len
        self.max_sent_id = max_sent_id


def extra_parsed_sentence_dict_from_org_file(org_file):
    rows = file_tool.load_data(org_file, 'r')
    parsed_sentence_dict = {}
    pb = progress_bar.SimpleProgressBar()
    print('begin extra parsed sentence dict from original file')
    count = len(rows)
    for row_index, row in enumerate(rows):
        items = row.strip().split('[Sq]')
        if len(items) != 4:
            raise ValueError("file format error")
        sent_id = str(items[0].strip())
        org_sent = str(items[1].strip())
        sent_tokens = str(items[2].strip()).split(' ')

        dependencies = []
        dep_strs = str(items[3].strip()).split('[De]')
        for dep_str in dep_strs:
            def extra_word_index(wi_str):
                wi_str_temp = wi_str.split('-')
                word = ''.join(wi_str_temp[:-1])
                index = str(wi_str_temp[-1])
                return word, index

            dep_itmes = dep_str.strip().split('[|]')
            if len(dep_itmes) != 3:
                raise ValueError("file format error")
            dep_name = str(dep_itmes[0]).strip()
            first_word, first_index = extra_word_index(str(dep_itmes[1]).strip())
            second_word, second_index = extra_word_index(str(dep_itmes[2]).strip())

            word_pair = {
                "first": {"word": first_word, "index": first_index},
                "second": {"word": second_word, "index": second_index}
            }

            dependency_dict = {
                'name': dep_name,
                'word_pair': word_pair
            }
            dependencies.append(dependency_dict)

        for w in sent_tokens:
            if w == '' or len(w) == 0:
                raise ValueError("file format error")

        parsed_info_dict = {
            'original': org_sent,
            'words': sent_tokens,
            'dependencies': dependencies,
            'id': sent_id,
            'has_root': True
        }
        if sent_id in parsed_sentence_dict:
            raise ValueError("file format error")

        parsed_sentence_dict[sent_id] = parsed_info_dict

        pb.update(row_index/count * 100)

    return parsed_sentence_dict
# def extra_parsed_sentence_dict_from_org_file(org_file):
#     rows = file_tool.load_data(org_file, 'r')
#     parsed_sentence_dict = {}
#     for row in rows:
#         items = row.strip().split('[Sq]')
#         if len(items) != 4:
#             raise ValueError("file format error")
#         sent_id = str(items[0].strip())
#         org_sent = str(items[1].strip())
#         sent_tokens = str(items[2].strip()).split(' ')
#
#         dependencies = []
#         dep_strs = str(items[3].strip()).split('[De]')
#         for dep_str in dep_strs:
#             def extra_word_index(wi_str):
#                 wi_str_temp = wi_str.split('-')
#                 word = ''.join(wi_str_temp[:-1])
#                 index = str(wi_str_temp[-1])
#                 return word, index
#
#             dep_itmes = dep_str.strip().split('[|]')
#             if len(dep_itmes) != 3:
#                 raise ValueError("file format error")
#             dep_name = str(dep_itmes[0]).strip()
#             first_word, first_index = extra_word_index(str(dep_itmes[1]).strip())
#             second_word, second_index = extra_word_index(str(dep_itmes[2]).strip())
#
#             word_pair = {
#                 "first": {"word": first_word, "index": first_index},
#                 "second": {"word": second_word, "index": second_index}
#             }
#
#             dependency_dict = {
#                 'name': dep_name,
#                 'word_pair': word_pair
#             }
#             dependencies.append(dependency_dict)
#
#         for w in sent_tokens:
#             if w == '' or len(w) == 0:
#                 raise ValueError("file format error")
#
#         parsed_info_dict = {
#             'original': org_sent,
#             'words': sent_tokens,
#             'dependencies': dependencies,
#             'id': sent_id,
#             'has_root': True
#         }
#         if sent_id in parsed_sentence_dict:
#             raise ValueError("file format error")
#
#         parsed_sentence_dict[sent_id] = parsed_info_dict
#
#     return parsed_sentence_dict

def group_dependencies(parsing_sentence_dict):
    unify_group_flag = True
    type_set = set()
    for sent_id, parsing_info in parsing_sentence_dict.items():
        for dependency in parsing_info['dependencies']:
            group = dependency2group_dict[dependency['name']]
            if unify_group_flag:
                group = "unify_group"
            dependency['name'] = group
            type_set.add(dependency['name'])
    if unify_group_flag:
        if len(type_set) != 1:
            raise ValueError
    else:
        if len(type_set) != group_count:
            raise ValueError
    return parsing_sentence_dict

def delete_root_dependency(parsing_sentence_dict):
    for sent_id, parsing_info in parsing_sentence_dict.items():
        root_index = -1
        for index, dependency in enumerate(parsing_info['dependencies']):
            name = dependency['name']
            if name == 'root':
                root_index = index
            else:
                first_index = dependency['word_pair']['first']['index']
                second_index = dependency['word_pair']['second']['index']
                if (not number_tool.is_number(first_index)) or (not number_tool.is_number(second_index)):
                    continue

                fw_i = int(first_index)
                dependency['word_pair']['first']['index'] = str(fw_i-1)

                sw_i = int(second_index)
                dependency['word_pair']['second']['index'] = str(sw_i-1)
        del parsing_info['dependencies'][root_index]
        if parsing_info['words'][0] != 'ROOT':
            raise ValueError
        del parsing_info['words'][0]
        parsing_info['has_root'] = False
    return parsing_sentence_dict

def modify_dependency_name(parsing_sentence_dict):
    for sent_id, parsing_info in parsing_sentence_dict.items():
        for dependency in parsing_info['dependencies']:
            name = dependency['name']
            name = name.split(':')
            if len(name) > 1:
                dependency['name'] = name[0]
    return parsing_sentence_dict


def process_parsing_sentence_dict(parsing_sentence_dict, modify_dep_name=False, delete_root_dep=False, group=True):
    if modify_dep_name:
        parsing_sentence_dict = modify_dependency_name(parsing_sentence_dict)

    if delete_root_dep:
        parsing_sentence_dict = delete_root_dependency(parsing_sentence_dict)

    if group:
        parsing_sentence_dict = group_dependencies(parsing_sentence_dict)

    word_dict = {}
    dependency_dict = {}

    error_dependencies_dict = {}
    count_of_dependencies = 0
    for sent_id, parsing_info in parsing_sentence_dict.items():
        for word in parsing_info['words']:
            if word in word_dict:
                word_info = word_dict[word]
                word_info['frequency'] += 1
            else:
                word_info = {'word': word, 'frequency': 1}
                word_dict[word] = word_info

        error_dependencies_list = []
        for dependency in parsing_info['dependencies']:
            count_of_dependencies += 1
            first_index = dependency['word_pair']['first']['index']
            second_index = dependency['word_pair']['second']['index']
            if (not number_tool.is_number(first_index)) or (not number_tool.is_number(second_index)):
                error_dependencies_list.append(dependency)
                continue
            if dependency['name'] in dependency_dict:
                dependency_info = dependency_dict[dependency['name']]
                dependency_info['frequency'] += 1
            else:
                dependency_info = {'dependency': dependency, 'frequency': 1}
                dependency_dict[dependency['name']] = dependency_info

        if len(error_dependencies_list) > 0:
            error_dependencies_dict[sent_id] = error_dependencies_list

    ##sort dependency
    dependency_dict = dict(sorted(dependency_dict.items(), key=lambda item: int(item[1]['frequency']), reverse=True))

    ##count error dep
    count_error_dependencies = 0
    for error_dependencies in error_dependencies_dict.values():
        count_error_dependencies += len(error_dependencies)

    count_temp = 0
    for dep in dependency_dict.values():
        count_temp += dep['frequency']
    if count_temp + count_error_dependencies != count_of_dependencies:
        raise ValueError
    print("The count of sentence: {}".format(len(parsing_sentence_dict)))
    print("The count of sentence including error_dependency: {}\terror_rate:{}".format(len(error_dependencies_dict),
                                                                                       round(len(
                                                                                           error_dependencies_dict) / len(
                                                                                           parsing_sentence_dict), 4)))
    print("The count of dependencies: {}\terror_rate:{}".format(count_of_dependencies,
                                                                round(count_error_dependencies / count_of_dependencies,
                                                                      4)))
    print("The count of error_dependencies: {}".format(count_error_dependencies))

    word_id_base = 0
    dependency_id_base = 0
    word_count = len(word_dict)
    dependency_count = len(dependency_dict)
    id2word_dict = {}
    id2dependency_dict = {}

    for i, word in enumerate(word_dict.keys(), word_id_base):
        word_info = word_dict[word]
        if str(i) in id2word_dict:
            raise ValueError
        word_info['id'] = i
        id2word_dict[str(i)] = word_info

    for i, dependency in enumerate(dependency_dict, dependency_id_base):
        dependency_info = dependency_dict[dependency]
        if str(i) in id2dependency_dict:
            raise ValueError
        dependency_info['id'] = i
        id2dependency_dict[str(i)] = dependency_info

    if len(id2word_dict) != word_count or len(id2dependency_dict) != dependency_count:
        raise RuntimeError

    dependency_save_data = ['\t'.join((item[0], str(item[1]['frequency']), str(item[1]['id']))) for item in dependency_dict.items()]
    dependency_save_data.insert(0, "\t".join(['name', 'frequency', 'id']))
    file_tool.save_list_data(dependency_save_data, 'data_preprocess/dependency.txt', 'w')

    ## numeral words and dependencies of sentences
    numeral_sentence_dict = {}
    error_parsing_sentence = []
    for sent_id, parsing_info in parsing_sentence_dict.items():
        words = parsing_info['words']
        dependencies = parsing_info['dependencies']
        sentence_len = len(words)

        numeral_words = []
        numeral_dependencies = []

        for word in words:
            numeral_words.append(int(word_dict[word]['id']))

        max_index = 0
        for dep_info in dependencies:
            dep_id = str(dependency_dict[dep_info['name']]['id'])
            word_pair = dep_info['word_pair']
            first_word = word_pair['first']
            second_word = word_pair['second']
            first_index = first_word['index']
            second_index = second_word['index']
            if (not number_tool.is_number(first_index)) or (not number_tool.is_number(second_index)):
                continue
            numeral_dependencies.append((int(first_index), int(second_index), int(dep_id)))

            if int(first_index) > max_index:
                max_index = int(first_index)
            if int(second_index) > max_index:
                max_index = int(second_index)
        if max_index > sentence_len - 1:
            error_parsing_sentence.append(parsing_info)

        numeral_sentence_dict[sent_id] = {
            'words': numeral_words,
            "dependencies": numeral_dependencies
        }
        # numeral_dict

    error_parsing_sentence_ids = [str(x["id"]) for x in error_parsing_sentence]
    if len(error_parsing_sentence_ids) > 0:
        print(error_parsing_sentence_ids)
        # file_tool.save_list_data(error_parsing_sentence_ids,
        #                          file_abs_path + '/proceess_files/error_parsing_sentence_ids.txt', 'w')
        raise ValueError

    return ParseInfo(numeral_sentence_dict=numeral_sentence_dict, dependency_dict=dependency_dict,
                     id2dependency=id2dependency_dict)


def dependencies2adj_matrix(numeral_dependencies, max_dep_kind, max_sent_len):
    result = np.zeros((max_dep_kind, max_sent_len, max_sent_len), dtype=np.int)
    numeral_dependencies = np.array(numeral_dependencies, dtype=np.int)
    dependencies_temp = numeral_dependencies[:, (2, 0, 1)]
    for dependency in dependencies_temp:
        result[tuple(dependency.reshape(-1, 1).tolist())] = 1
    return result


