from src.models.dqn_injected_plasticity_model import DQNInjectedPlasticityModel
from src.models.dqn_model import get_model_duelling_dqn
from src.models.model_with_injected_plasticity_2 import get_model_injected_plasticity


def get_model(model_type, num_classes, seed, input_shape, eta, alpha):
    if model_type == 'dqn':
        return get_model_duelling_dqn(num_classes, seed, input_shape)
    elif model_type == 'dqn_injected_plasticity':
        return DQNInjectedPlasticityModel(num_classes, seed, input_shape, eta, alpha)
    else:
        raise Exception("Model not implemented")
