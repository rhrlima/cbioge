import keras.layers

from .layers import ResBlock

def _get_layer(blocK_name, layers_database=[keras.layers]):
    for db in layers_database:
        try:
            return getattr(db, blocK_name)
        except Exception as e:
            pass

    raise AttributeError(f'{blocK_name} not found in the database')