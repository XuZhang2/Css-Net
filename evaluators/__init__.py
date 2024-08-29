from evaluators.tirg_evaluator import SimpleEvaluator
from evaluators.tirg_evaluator import SimpleEvaluator_cosmo


def get_evaluator_cls(configs):
    evaluator_code = configs['evaluator']
    if evaluator_code == 'simple':
        return SimpleEvaluator
    elif evaluator_code == 'cosmo':
        return SimpleEvaluator_cosmo
    else:
        raise ValueError("There's no evaluator that has {} as a code".format(evaluator_code))
