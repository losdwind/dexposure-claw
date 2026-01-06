#!/bin/bash
cd /home/figurich/inter-protocol-exposure/graphpfn
CUDA_VISIBLE_DEVICES="" .venv/bin/python bin/evaluate.py exp/graphpfn-eval/finetune/raw/dexposure-quick/evaluation.toml --force 2>&1
