#!/bin/bash

model=flan-t5-base
case_id=200

python src/train_peft.py --model $model \
                         --case_id $case_id
