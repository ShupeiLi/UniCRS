#!/bin/bash
sh scripts/train_pre_inspired.sh &&\
sh scripts/train_conv_inspired.sh &&\
sh scripts/infer_conv_inspired.sh &&\
sh scripts/train_rec_inspired.sh
