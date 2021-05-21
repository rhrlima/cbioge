class BaseProblem:

    def map_genotype_to_phenotype(self, solution) -> str:
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, phenotype):
        raise NotImplementedError('Not implemented yet.')

class DNNProblem(BaseProblem):

    def __init__(self, parser, dataset):
        self.parser = parser
        self.dataset = dataset

    def predict(self, model, weights=None):
        ''' runs the prediction on a model
            if weights are provided, the model will feed them into the network

            return: list structure with the predictions of the network for the
            given dataset
        '''
        raise NotImplementedError('Not implemented yet.')