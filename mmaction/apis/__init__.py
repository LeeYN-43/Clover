from .inference import inference_recognizer, init_recognizer
from .test import multi_gpu_test, single_gpu_test
from .train import train_model

try:
    from .train_prefetch import prefetch_train_model
except Exception as e:
    print(e)

__all__ = [
    'train_model', 'init_recognizer', 'inference_recognizer', 'multi_gpu_test',
    'single_gpu_test'
]
