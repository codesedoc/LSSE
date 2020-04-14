import framework as fr
import argparse
import utils.hyperor as hyperor
import utils.general_tool as general_tool
import utils.file_tool as file_tool
import torch
import logging
# import analysis.error_analysis as er_analysis
# import analysis.mrpc_analysis as mrpc_analysis
from model import MODEL_CLASSES, ALL_MODELS
from glue.glue_manager import glue_processors as processors
import os

def create_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "-td",
        "--transformer_dropout",
        type=float,
        default=0,
        help="The probability of hidden layer of transformer.",
    )

    parser.add_argument(
        "-gd",
        "--gcn_dropout",
        type=float,
        default=0.1,
        help="The probability of hidden layer of gcn.",
    )

    parser.add_argument(
        "-f",
        "--framework_name",
        type=str,
        default="SeE",
        help="The name of framework, choose in [SeE, LE, LSeE, LSyE, LSSE]",
        choices=['SeE', 'LE', 'LSeE', 'LSyE', 'LSSE']
    )

    parser.add_argument(
        "-scf",
        "--semantic_compare_func",
        type=str,
        default="l2",
        help="The name of framework, choose in [l2, wmd]",
        choices=['l1', 'l2', 'wmd']
    )

    parser.add_argument(
        "-woc",
        "--without_concatenate_input_for_gcn_hidden",
        action="store_true",
        help="Whether without_concatenate_input_for_gcn_hidden?",
    )

    parser.add_argument(
        "-gl",
        "--gcn_layer",
        type=int,
        default=2,
        help="The name of gcn layer",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )



    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
            file_tool.check_dir(args.output_dir)
            and file_tool.list_dir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.transformer_dropout < 0 or args.transformer_dropout > 1:
        raise ValueError
    args.transformer_dropout = round(args.transformer_dropout, 1)

    if args.gcn_dropout < 0 or args.gcn_dropout > 1:
        raise ValueError
    args.gcn_dropout = round(args.gcn_dropout, 1)

    args.task_name = args.task_name.lower()
    args.device = device
    args.max_sentence_length = 50
    args.optimizer = 'adam'
    args.framework_with_gcn = ['LSSE', 'LSyE']
    args.encoder_hidden_dim = 768
    args.fully_scales = [args.encoder_hidden_dim * 2, 2]
    args.max_encoder_seq_length = args.max_seq_length
    if args.framework_name in args.framework_with_gcn:
        args.gcn_hidden_dim = args.encoder_hidden_dim
        args.gcn_gate_flag = True
        args.gcn_norm_item = 0.5
        args.gcn_self_loop_flag = True
        args.gcn_group_layer_limit_flag = False
        if args.gcn_group_layer_limit_flag:
            args.gcn_dep_layer_limit_list = [6, 5, 4, 3, 2]
        args.gcn_position_encoding_flag = True

        args.fully_scales = [args.gcn_hidden_dim * 2, 2]

        if not args.without_concatenate_input_for_gcn_hidden:
            args.fully_scales[0] += args.gcn_hidden_dim
    else:
        del args.gcn_layer
    return args

def check_dropout(model):
    result = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            result.append(m)
    return result

def run_framework():
    # raise ValueError('my error!')
    args = create_args()

    framework_manager = fr.FrameworkManager(args)
    # dropouts = check_dropout(framework_manager.framework)
    # framework_manager.train_model()
    framework_manager.run()
    # framework_manager.visualize_model()


def run_hyperor():
    args = create_args()

    hyr = hyperor.Hyperor(args)
    hyr.tune_hyper_parameter()


# def corpus_test():
#     # corpus.stsb.test()
#     # corpus.mrpc.test()
#     # mrpc_obj = corpus.mrpc.get_mrpc_obj()
#     # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
#     # general_tool.covert_transformer_tokens_to_words(mrpc_obj, tokenizer,
#     #                                                 'corpus/mrpc/sentence_words(bert-base-cased).txt',
#     #                                                 '##')
#     # qqp_obj = corpus.qqp.get_qqp_obj()
#     # tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
#     # general_tool.covert_transformer_tokens_to_words(qqp_obj, tokenizer,
#     #                                                 'corpus/qqp/sentence_words(bert-base-cased).txt',
#     #                                                 '##')
#     # general_tool.calculate_the_max_len_of_tokens_split_by_bert(qqp_obj, tokenizer)
#     # corpus.qqp.test()
#
#     stsb_obj = corpus.stsb.get_stsb_obj()
#     tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
#     general_tool.calculate_the_max_len_of_tokens_split_by_bert(stsb_obj, tokenizer)
#     # general_tool.covert_transformer_tokens_to_words(stsb_obj, tokenizer,
#     #                                                 'corpus/stsb/sentence_words(bert-base-cased).txt',
#     #                                                 '##')

def main():

    # run_framework()
    run_hyperor()
    # er_analysis.test()
    # mrpc_analysis.test()
    # corpus_test()


def occupy_gpu():
    memory = []
    while(True):
        try:
            memory.append(torch.zeros((40,)*4, dtype=torch.double, device=torch.device('cuda', 0)))
        except Exception as e:
            print(e)
            break
    return memory

if __name__ == '__main__':

    try:
        main()
        # raise ValueError
    except Exception as e:
        raise
        # logging.exception(e)
        try:
            memory = occupy_gpu()
            print('finish apply memory')
            if len(memory) >1:
                del memory[len(memory) - 1]

            if len(memory) == 0:
                m = torch.zeros((10,), dtype=torch.int, device=torch.device('cuda', 0))
            else:
                m = memory[0]

            while (True):
                m * m
        except Exception as e:
            m = torch.zeros((10,)*4, dtype=torch.int, device=torch.device('cpu'))
            print('defeat occupy !')
            while(True):
                m * m


    pass