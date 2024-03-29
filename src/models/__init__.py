from src.models.model import get_model_static_frame


def get_model(model_type, input_shape, num_classes, seed):
    if model_type == 'static_frame':
        return get_model_static_frame(input_shape, num_classes, seed)
    else:
        raise Exception("Model not implemented")
