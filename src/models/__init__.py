from src.models.dqn_model import get_model_duelling_dqn
from src.models.model_with_injected_plasticity import get_model_injected_plasticity


def get_model(model_type, num_classes, seed, input_shape, eta, alpha):
    if model_type == 'dqn':
        return get_model_duelling_dqn(num_classes, seed, input_shape)
    elif model_type == 'dqn_injected_plasticity':
        return get_model_injected_plasticity(num_classes, seed, input_shape, eta, alpha)
    else:
        raise Exception("Model not implemented")
