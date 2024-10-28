#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# export CUDA_HOME=$CONDA_PREFIX # change to your path
# export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
# export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
# TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME='/path/to/your/cuda/dir'  
python setup.py build install
