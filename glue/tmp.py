import sys
sys.path.insert(0, '/home/sheng/Documents/study/workspace/python/PIRs')

import utils.file_tool as file_tool
import utils.general_tool as general_tool


def update_id_of_parsed_sentences(org_sent_file, old_file, new_file, split_str):
    sents = file_tool.load_data(org_sent_file, 'r')
    org_sent_id_dict = {}
    for line in sents:
        items = line.split('\t')
        if len(items) != 2:
            raise ValueError
        org_sent_id_dict[items[1].strip()] = items[0]

    rows = file_tool.load_data(old_file, 'r')
    save_data = []
    no_find = []
    for r in rows:
        items = r.split(split_str)
        if len(items) != 4:
            raise ValueError
        if not general_tool.is_number(items[0]):
            raise ValueError
        if items[1] not in org_sent_id_dict:
            no_find.append(items[1])
        else:
            items[0] = org_sent_id_dict[items[1]]
        save_data.append(split_str.join(items))
    print('can not find {} original sentence'.format(len(no_find)))
    file_tool.save_list_data(save_data, new_file, 'w')

if __name__ == '__main__':
    org_sent_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/original_sentence.txt'
    old_parsed_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/parsed_sentences_old.txt'
    new_parsed_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/parsed_sentences.txt'
    update_id_of_parsed_sentences(org_sent_file, old_parsed_file, new_parsed_file, '[Sq]')