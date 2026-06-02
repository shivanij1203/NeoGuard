"""N-CNN: clean-room reimplementation of the Zamzmi et al. (IJCNN 2019)
neonatal pain CNN, per the project's NCNN_implementation_spec.md.

Research prototype, not a medical device. Do not interpret model output as
a clinical assessment.
"""
from ml.ncnn.model import NCNN

__all__ = [
    "NCNN",
]
