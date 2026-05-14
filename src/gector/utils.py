import inspect
from transformers import AutoModel

_ADD_POOLING_CACHE = {}

def has_args_add_pooling(model_id: str) -> bool:
    if model_id in _ADD_POOLING_CACHE:
        return _ADD_POOLING_CACHE[model_id]
    import inspect
    from transformers import AutoModel
    sig = inspect.signature(AutoModel.from_config(
        __import__('transformers').AutoConfig.from_pretrained(model_id)
    ).__class__.__init__)
    result = 'add_pooling_layer' in sig.parameters
    _ADD_POOLING_CACHE[model_id] = result
    return result
