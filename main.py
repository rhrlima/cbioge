import sys

from experiments.unet_experiment import run_experiment
from experiments.unet_evolution import run_evolution


if 'exp' in sys.argv:
    sys.argv.remove('exp')
    run_experiment()

elif 'evo' in sys.argv:
    sys.argv.remove('evo')
    run_evolution()


# TODO revisar metricas
# TODO revisar datasets
# TODO re executar evolução