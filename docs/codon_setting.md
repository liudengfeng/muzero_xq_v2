# codon 配置

## Python integration

Calling Python from Codon is possible in two ways:

+ from python import allows importing and calling Python functions from existing Python modules.
+ @python allows writing Python code directly in Codon.
  
In order to use these features, the CODON_PYTHON environment variable must be set to the appropriate Python shared library:

```bash
# 案例
export CODON_PYTHON=/path/to/libpython.X.Y.so
# 本机
export CODON_PYTHON=/home/ldf/anaconda3/envs/rl/lib/libpython3.10.so
```
### from python import

Let's say we have a Python function defined in mymodule.py:

```python
def multiply(a, b):
    return a * b
```

We can call this function in Codon using from python import and indicating the appropriate call and return types:

```codon
from python import mymodule.multiply(int, int) -> int
print(multiply(3, 4))  # 12
```

(Be sure the PYTHONPATH environment variable includes the path of mymodule.py!)

```bash
export PYTHONPATH=$(pwd)
```

## GPU

```bash
# 编译
--libdevice=/usr/local/cuda-12.1/nvvm/libdevice/libdevice.10.bc
--libdevice="/mnt/C/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/nvvm/libdevice/libdevice.10.bc"
```