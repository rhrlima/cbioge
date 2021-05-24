class BaseProblem:

    def map_genotype_to_phenotype(self, solution) -> str:
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, phenotype):
        raise NotImplementedError('Not implemented yet.')

class DNNProblem(BaseProblem):

    def __init__(self, parser, dataset):
        self.parser = parser
        self.dataset = dataset

    def _set_data_size(value, target):
        if type(value) is float and 0 < value <= 1:
            # percentage
            return int(value * target)
        elif type(value) is int and value >= 1:
            # absolute
            return min(value, target)
        else:
            # unknown
            return target

    def set_data_size(self, train=1.0, valid=1.0, test=1.0):
        ''' Sets the portions of the dataset that will be used in training, 
            validation, and test. Values [0, 1] are considered as % of the 
            dataset (floor), and absolute values will be used as is.
        '''
        x_train = self.x_train[:self.train_size]
        y_train = self.y_train[:self.train_size]
        x_valid = self.x_valid[:self.valid_size]
        y_valid = self.y_valid[:self.valid_size]
        x_test = self.x_test[:self.test_size]
        y_test = self.y_test[:self.test_size]

    def predict(self, model, weights=None):
        ''' runs the prediction on a model
            if weights are provided, the model will feed them into the network

            return: list structure with the predictions of the network for the
            given dataset
        '''
        raise NotImplementedError('Not implemented yet.')