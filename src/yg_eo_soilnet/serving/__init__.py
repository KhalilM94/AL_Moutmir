"""Loading a trained deep-learning model and predicting with it, outside a training run.

The sklearn path has had this for free: ``mlflow.sklearn.log_model`` serializes a whole fitted
``Pipeline``, so the imputers, the scaler and the encoder travel with the estimator and a reloaded
model can be handed raw data. The Lightning path had only half of it - the checkpoint carried the
weights and the target inverse-transform, but the INPUT standardization and the categorical
vocabulary were fitted on the datamodule and thrown away with it, so a restored model could not be
fed anything it had not already been fed.

:meth:`SoilSequenceDataModule.preprocessing_state` and
:meth:`SoilRegressionLightningBase.attach_preprocessing_state` close that gap by putting the fitted
statistics in the checkpoint; :class:`SoilSequencePredictor` is what uses them.
"""

from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

__all__ = ["SoilSequencePredictor"]
