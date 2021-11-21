from cbioge.problems import BaseProblem, CNNProblem, DNNProblem


def test_is_cnnproblem_subclass_of_problem():
    assert issubclass(CNNProblem, BaseProblem)

def test_is_dnnproblem_subclass_of_problem():
    assert issubclass(DNNProblem, BaseProblem)
