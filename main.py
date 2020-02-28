import corpus
import framework as fr
import argparse
import utils.hyperor as hyperor
import utils.general_tool as general_tool
import torch
import logging


def create_arg_dict():

    arg_dict = {
        'batch_size': 8,
        'learn_rate': 2e-5,
        # 'sgd_momentum': 0.4,
        'optimizer': 'adam',
        'k_fold': 10,
        'epoch': 3,
        'warmup_steps': 0,
        'max_steps': -1,
        'gcn_layer': 2,
        'position_encoding': True,
        'dropout': 0.4,
        'regular_flag': False,
        'ues_gpu': -1,
        'repeat_train': True,
        'corpus': corpus.mrpc.get_mrpc_obj,
        # 'max_sentence_length': 50,
        'framework_name': "LSSE",
        'task_type': 'classification',
        'seed': 1234
    }
    general_tool.setup_seed(arg_dict['seed'])
    parser = argparse.ArgumentParser(description='PIRs')
    parser.add_argument('-gpu', dest="ues_gpu", default='0', type=int,
                        help='GPU order, if value is -1, it use cpu. Default value 0')

    args = parser.parse_args()
    args = vars(args)
    arg_dict.update(args)

    return arg_dict


def run_framework():
    # raise ValueError('my error!')
    arg_dict = create_arg_dict()
    framework_manager = fr.FrameworkManager(arg_dict)
    # framework_manager.train_model()
    framework_manager.train_final_model()
    framework_manager.test_model()
    # framework_manager.visualize_model()


def run_hyperor():
    arg_dict = create_arg_dict()
    hyr = hyperor.Hyperor(arg_dict)
    hyr.tune_hyper_parameter()


def main():
    # corpus.mrpc.get_mrpc_obj()
    corpus.qqp.test()
    # run_framework()
    # run_hyperor()


def occupy_gpu():
    memory = []
    while(True):
        try:
            memory.append(torch.zeros((10,)*4, dtype=torch.double, device=torch.device('cuda', 0)))
        except Exception as e:
            print(e)
            break
    return memory

if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        logging.exception(e)
        memory = occupy_gpu()
        while(True): pass

    pass