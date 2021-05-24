import numpy as np

from keras.models import Model
from keras.layers import Input, Dense

from cbioge.problems.dnn import ModelRunner


def get_valid_mockup_model():
    inputs = Input([256])
    dense1 = Dense(256)(inputs)
    dense2 = Dense(10)(dense1)
    model = Model(input=inputs, output=dense2)
    model.compile('Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_train_model():
    try:
        x_train = np.zeros((10,256))
        y_train = np.zeros((1,10))
        mockup_model = get_valid_mockup_model()
        runner = ModelRunner(mockup_model)
        runner.train_model(x_train, y_train, 1, 1)
        flag = True
    except Exception as e:
        print(e)
        flag = False
    finally:
        assert flag == True



# @pytest.mark.parametrize(
#     "test_input,expected",
#     [("3+5", 8), ("2+4", 6), pytest.param("6*9", 42, marks=pytest.mark.xfail)],
# )
# def test_eval(test_input, expected):
#     assert eval(test_input) == expected