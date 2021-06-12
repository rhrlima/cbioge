from keras.optimizers import get
import numpy as np

import pytest

from keras.models import Model
from keras.layers import Input, Dense

from cbioge.problems.dnn import ModelRunner
from cbioge.utils import checkpoint as ckpt


def get_valid_mockup_model():
    inputs = Input([256])
    dense1 = Dense(256)(inputs)
    dense2 = Dense(10)(dense1)
    model = Model(input=inputs, output=dense2)
    model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_isinstance_of_modelrunner():
    runner = ModelRunner(None)
    assert isinstance(runner, ModelRunner)

@pytest.mark.parametrize('model', [None, get_valid_mockup_model()])
@pytest.mark.parametrize('path', [None, ckpt.ckpt_folder, 'test_folder'])
@pytest.mark.parametrize('verbose', [True, False, 0, 1])
def test_modelrunner_default_params2(model, path, verbose):
    runner = ModelRunner(model, path=path, verbose=verbose)
    assert runner.model is model
    assert runner.loss == 1
    assert runner.accuracy == 0

    params_ref = model.count_params() if model is not None else 0
    assert runner.params == params_ref
    assert runner.history is None
    assert runner.verbose is verbose

    path_ref = ckpt.ckpt_folder if path is None else path
    assert runner.ckpt_path == path_ref
