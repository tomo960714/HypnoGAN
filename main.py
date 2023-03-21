# -*- coding: UTF-8 -*-
# Local packages:
import argparse
import logging
import os
import pickle
import random
import shutil
import time

# 3rd party packages:
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
# TODO: Implement Neptune logger

# personal packages:
from Data.preprocess import preprocess_data
from model.timegan import TimeGAN
from model.utils import timegan_trainer, timegan_generator

# TODO: START TRAINING!!!

def main(args):
    ######################################
    # Initialize output directories
    ######################################

    ## runtime directory
    code_dir = os.path.abspath(".")
    if not os.path.exists(code_dir):
        raise ValueError(f"Code directory not found at {code_dir}")
    
    ## Data directory
    data_path = os.path.abspath("./Data")
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory not found at {data_path}")
    data_dir = os.path.dirname(data_path)
    data__file_name = os.path.basename(data_path)

    ## Output directory
    args.model_path = os.path.abspath(f"./Output/{args.exp}/")
    out_dir = os.path.abspath(args.model_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exists_ok=True)
    
    ## Tensorboard directory
    tb_path = os.path.abspath("./tensorboard")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path,exists_ok=True)
    
    print(f"Code directory:\t\t\t{code_dir}")
    print(f"Data directory:\t\t\t{data_path}")
    print(f"Output directory:\t\t{out_dir}")
    print(f"Tensorboard directory:\t\t{tb_path}\n")

    ######################################
    # Initialize random seed and CUDA
    ######################################
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    ######################################
    # Load and preprocess data for model
    ######################################
    X,T,loaded_data_info = preprocess_data(data_limit=args.data_limit)

    print(f"Processed Data: {X.shape} (Idx x Max_Sequence_Length x Features(=1))")
    print(f"Original data preview:\n{X[:2, :10, :2]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    args.max_seq_len = loaded_data_info.max_length
     # Train-Test Split data and time
     # TODO: Same people should be in the same pool at train test split
     # Make the split on the subject ID's, so also have to att ID to the loaded data
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, test_size=args.train_rate, random_state=args.seed
    )
    #########################
    # Initialize and Run model
    #########################

     # Log start time
    start = time.time()

    model = TimeGAN(args)
    if args.is_train == True:
        timegan_trainer(model, train_data, train_time, args)
    generated_data = timegan_generator(model, train_time, args)
    generated_time = train_time
    # Log end time
    end = time.time()
    
    print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start)/60} mins\n")

    # Save splitted data and generated data
    with open(f"{args.model_path}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.model_path}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.model_path}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.model_path}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)

    #TODO: metrics

    return None

if __name__ == "__main__":
    args = (
        device = 'cuda'
        exp = 'test'
        is_train = True
        seed = 0
        data_limit = 10
        train_rate = 0.6
        max_seq_len = 100
        
    )
    main(args)
"""
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

   # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
        type=int)

    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)
    parser.add_argument(
        '--data_limit',
        default=10,
        type=int)

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float)

    args = parser.parse_args()

    # Call main function
    main(args)

"""
