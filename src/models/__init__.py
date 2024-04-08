from src.models.dqn_injected_plasticity_model import DDQNInjectedPlasticityModel
from src.models.ddqn_model import get_model_duelling_dqn
from src.models.dqn_model import get_model_dqn


def get_model(model_type, num_classes, seed, input_shape, eta, alpha):
    if model_type == "dqn":
        return get_model_dqn(num_classes, seed, input_shape)
    elif model_type == 'ddqn':
        return get_model_duelling_dqn(num_classes, seed, input_shape)
    elif model_type == 'ddqn_injected_plasticity':
        return DDQNInjectedPlasticityModel(num_classes, seed, input_shape, eta, alpha)
    else:
        raise Exception("Model not implemented")
