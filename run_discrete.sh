#!/bin/bash

model=flan-t5-base
case_id=0

python src/run_discrete.py --model $model \
                           --case_id $case_id
