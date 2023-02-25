#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='the client rank')
    parser.add_argument('--rounds',
                        default=100,
                        type=int,
                        help='total communication rounds')
    parser.add_argument('--epoch',
                        default=50,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='local batch size')
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='learning rate')
    parser.add_argument('--seed',
                        default=3456,
                        type=int,
                        help='random seed')
    parser.add_argument('--label_owner_address',
                        default='127.0.0.1:59290',
                        type=str,
                        help='init method')
    parser.add_argument('--server_address',
                        default='127.0.0.1:59291',
                        type=str,
                        help='init method')
    parser.add_argument('--num_clients',
                        default=2,
                        type=int,
                        help='local_count/sum_count')
    parser.add_argument('--n_features',
                        default=12,
                        type=int,
                        help='the number of features')
    parser.add_argument('--ctx_file',
                        default='../../transmission/ts_ckks.config',
                        type=str,
                        help='the number of features')
    args = parser.parse_args()
    return args
