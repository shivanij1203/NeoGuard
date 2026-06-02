"""N-CNN: clean-room reimplementation of the Zamzmi et al. (IJCNN 2019)
neonatal pain CNN, per the project's NCNN_implementation_spec.md.

Research prototype, not a medical device. Do not interpret model output as
a clinical assessment.

This package is built up over the following phases: the model, then the
calibration and inference path, then preprocessing, then the streaming and
fusion wiring. It starts empty by design.
"""
