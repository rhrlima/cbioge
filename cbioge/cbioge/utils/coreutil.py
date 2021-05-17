import os, shutil

def contains(small, big):
	items = set(big)
	return set(small).issubset(items)

def limit_gpu_memory(fraction=0.5):
    import tensorflow as tf
    from tensorflow.compat.v1.keras.backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.compat.v1.Session(config=config))

def create_dir(dir_path, override_if_exists=False):
	if not os.path.exists(dir_path):
		os.makedirs(f"{dir_path}")
	else:
		if override_if_exists:
			shutil.rmtree(dir_path)
			os.makedirs(f"{dir_path}")




