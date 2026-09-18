# Legacy training scripts — do not use

These trained on *synthetic feature vectors* (three Gaussian blobs: small / medium / large code, label = blob), so the models learned
"bigger is riskier" and reported meaningless 100 % accuracy. They also labelled Juliet by file name, which is mostly wrong.

They are kept only for reference. Use `ml/juliet_extract.py` -> `ml/build_features.py` -> `ml/retrain_all.py` (or `ml/bootstrap.py`).
