"""
This file specifies how MLC's Whisper parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .marian_model import MarianConfig, MarianMT, MAX_VOCAB_SIZE
from .marian_quantization import awq_quant


def pad_bias(bias, dtype):

    """Pad bias tensor to match MAX_VOCAB_SIZE.

    Parameters
    ----------
    bias : numpy.ndarray
        The bias tensor to pad
    dtype : numpy.dtype
        The target dtype for the padded tensor

    Returns
    -------
    numpy.ndarray
        The padded bias tensor with shape (1, MAX_VOCAB_SIZE) and specified dtype.
        Padding values are set to -inf to ensure their probs will be 0 after softmax.
    """
    
     
    padding = MAX_VOCAB_SIZE - bias.shape[1]
    # Create a new array with the padded shape, filled with the padding value (-inf in this case for softmax)
    bias = np.pad(bias, ((0, 0), (0, padding)), 'constant', constant_values=float("-inf"))
    return bias.astype(dtype)

def pad_weights(param, dtype):
    """Pad weight tensor to match MAX_VOCAB_SIZE.

    Parameters
    ----------
    param : numpy.ndarray
        The weight tensor to pad
    dtype : numpy.dtype
        The target dtype for the padded tensor

    Returns
    -------
    numpy.ndarray
        The padded weight tensor with shape (MAX_VOCAB_SIZE, *) and specified dtype.
        Padding values are set to 0 to avoid affecting the model outputs.
    """
     
    padding = MAX_VOCAB_SIZE - param.shape[0]
    param = np.pad(param, ((0, padding), (0, 0)), 'constant', constant_values=0)
    return param.astype(dtype)


def huggingface(model_config: MarianConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : WhisperConfig
        The configuration of the GPT-2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = MarianMT(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    
    mlc_name = ["lm_head.weight"]
    src_name = ["model.shared.weight"]
    for mlc_name, src_name in zip(mlc_name, src_name):
        mapping.add_mapping(
            mlc_name,
            [src_name],
            functools.partial(
                pad_weights,
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            map_func = functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                )
            if mlc_name in [ "model.encoder.embed_tokens.weight","model.shared.weight", "model.decoder.embed_tokens.weight", 'lm_head.weight']:
                map_func = functools.partial(
                    pad_weights,
                    dtype=mlc_param.dtype,
                )
            elif mlc_name == "final_logits_bias":
                 map_func = functools.partial(
                    pad_bias,
                    dtype=mlc_param.dtype,
                )
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                map_func,
            )

    return mapping


def awq(model_config: MarianConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : WhisperConfig
        The configuration of the Whisper model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    
    mlc_name = ["lm_head.weight"]
    src_name = ["model.shared.weight"]
    for mlc_name, src_name in zip(mlc_name, src_name):
        mapping.add_mapping(
            mlc_name,
            [src_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )
      

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            mapping.add_mapping(
                mlc_name,
                [mlc_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=mlc_param.dtype),
            )
    return mapping
