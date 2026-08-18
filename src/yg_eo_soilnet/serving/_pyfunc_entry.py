"""The models-from-code entry point MLflow stores and executes to rebuild a sequence model.

This file is passed to ``mlflow.pyfunc.log_model`` as ``python_model=<this path>``, which is the
form MLflow's own documentation marks *"(recommended)"*. The alternative - handing it a live Python
object - CloudPickles the whole object graph, which is how a trained CNN previously ended up as a
15 MB ``python_model.pkl`` whose weights could not be read without unpickling everything, and which
MLflow warns "can execute arbitrary code during deserialization".

What gets stored instead is this script plus one artifact: ``torch_model``, a real
``mlflow.pytorch`` model directory. That nesting is what makes the network recognisable as PyTorch -
``mlflow.pytorch.load_model`` works against ``<model_uri>/artifacts/torch_model`` and its own
MLmodel carries the ``pytorch`` flavor - while the outer model keeps the pyfunc contract that
accepts raw data. One deployable model, the weights stored once.

Note what this does NOT claim: the nested model is saved with ``serialization_format="pickle"``,
because ``pt2`` traces ``forward`` from a tensor example and this architecture consumes a dict batch
of ragged sequences. So an object pickle remains, scoped to the LightningModule. The run's own
``checkpoints/best.ckpt`` is the safe ``weights_only``-loadable copy.

MLflow executes this module at load time and takes whatever ``set_model`` was given, so the
construction below runs once per load rather than being frozen into a pickle.
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import mlflow.pytorch

from yg_eo_soilnet.serving.lightning_pyfunc import SoilSequencePyfunc

TORCH_MODEL_ARTIFACT = "torch_model"


class SoilSequenceEntry(SoilSequencePyfunc, mlflow.pyfunc.PythonModel):
    """Loads the nested PyTorch model, then predicts through the raw-data wrapper."""

    def load_context(self, context) -> None:
        # The nested model carries its own class, so nothing here has to name the architecture.
        # It also restores `preprocessing_state` - the fitted scalers and the categorical
        # vocabulary - which is what lets the wrapper standardize raw input.
        self.model = mlflow.pytorch.load_model(
            context.artifacts[TORCH_MODEL_ARTIFACT],
            map_location="cpu",
        )
        self.model.eval()
        self._predictor = None


mlflow.models.set_model(SoilSequenceEntry())
