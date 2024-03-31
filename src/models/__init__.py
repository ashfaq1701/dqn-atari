from src.models.model import get_model_basic_cnn


def get_model(model_type, num_classes, seed, input_shape):
    if model_type == 'dueling_dqn':
        return get_model_basic_cnn(num_classes, seed, input_shape)
    else:
        raise Exception("Model not implemented")
