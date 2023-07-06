If you have CUDA 11.6 or newer (display via `nvcc --version`), simply follow installation instructions in the main README.

If you have an older version of CUDA, you may not be able to use a [torch version](https://pytorch.org/get-started/previous-versions/) supported by peft. To circumnavigate this, clone the [peft repo](https://github.com/huggingface/peft) and replace their `setup.py` with our modified version in `env/peft-setup.py`. Then perform the following steps:

```
conda create -n radadapt python=3.8
conda activate radadapt
pip install ../peft # install local version of peft
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
pip install tensorboard datasets
pip install evaluate f1chexbert radgraph rouge_score bert_score # to calculate metrics
```
