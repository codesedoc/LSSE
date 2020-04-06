import utils.file_tool as file_tool
import utils.general_tool as general_tool


def update_id_of_parsed_sentences(org_sent_file, old_file, new_file, split_str):
    sents = file_tool.load_data(org_sent_file, 'r')
    org_sent_id_dict = {}
    for line in sents:
        items = line.split('\t')
        if len(items) != 3:
            raise ValueError
        org_sent_id_dict[items[1]] = items[0]

    rows = file_tool.load_data(old_file, 'r')
    save_data = []
    for r in rows:
        items = r.split(split_str)
        if items != 3:
            raise ValueError
        if not general_tool.is_number(items[0]):
            raise ValueError
        items[0] = org_sent_id_dict[items[1]]
        save_data.append(split_str.join(items))

    file_tool.save_list_data(save_data, new_file, 'w')

if __name__ == '__main__':
    import sys
    sys.path.append('/home/sheng/Documents/study/workspace/python/PIRs')
    org_sent_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/original_sentence.txt'
    old_parsed_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/parsed_sentences_old.txt'
    new_parsed_file = '/home/sheng/Documents/study/workspace/python/PIRs/glue/data/MRPC/parsed_sentences.txt'
    update_id_of_parsed_sentences(org_sent_file, old_parsed_file, new_parsed_file)