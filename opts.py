"""Argument parser"""

import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='./data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='./data/caches',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='./runs/vg/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='./runs/vg/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--logger_path', default='/runs/vg/log',
                        help='Path to save logs.')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=150, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=75, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=4e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=10, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--negation', type=int, default=0)

    # ------------------------- model setting -----------------------#
    parser.add_argument('--max_turns', default=10, type=int,
                        help='max rounds for queries.')
    parser.add_argument('--cross_attention_direction', default='I-T',
                        help='cross attention directions for local-level matching, i.e., I-T or T-I')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=256, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='normalize the text embeddings.')
    parser.add_argument('--icr_step', default=3, type=int,
                        help='the number of steps of the inter-correlation reasoning.')
    parser.add_argument('--EOS_idx', type=int, default=2)  
    parser.add_argument('--vocab_size', type=int, default=14284)         
    parser.add_argument('--model_path', type=str, default=None)    
    parser.add_argument('--temperature_lambda_i2t', type=float, default=5,
                        help='temperature coefficient for I-T cross attention') 
    parser.add_argument('--temperature_lambda_t2i', type=float, default=15,
                        help='temperature coefficient for T-I cross attention') 
    parser.add_argument('--seed', type=int, default=0)  
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='balance the weight of different losses')    
    parser.add_argument('--beta', type=float, default=0.3,
                        help='balance the weight of different losses')  
    parser.add_argument('--pin_memory', type=str2bool, default=True)     
    parser.add_argument('--tau', type=float, default=40,
                        help='temperature coefficient for InfoNCE loss')          
    opt = parser.parse_args()
    return opt
