#!/bin/bash
sh scripts/train_pre_redial.sh &&\
sh scripts/train_conv_redial.sh &&\
sh scripts/infer_conv_redial.sh &&\
sh scripts/train_rec_redial.sh
