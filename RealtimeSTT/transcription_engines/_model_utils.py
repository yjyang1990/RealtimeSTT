from collections.abc import Mapping


def attr_or_key(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def first_item(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def text_from_output(output):
    output = first_item(output)
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()

    text = attr_or_key(output, "text")
    if text is not None:
        return str(text).strip()

    nested_outputs = attr_or_key(output, "outputs")
    nested_output = first_item(nested_outputs)
    nested_text = attr_or_key(nested_output, "text")
    if nested_text is not None:
        return str(nested_text).strip()

    return str(output).strip()


def language_from_output(output, fallback=None):
    output = first_item(output)
    return attr_or_key(output, "language", fallback)


def move_to_device(value, device=None, dtype=None):
    if not hasattr(value, "to"):
        return value

    if device is not None and dtype is not None:
        try:
            return value.to(device, dtype=dtype)
        except TypeError:
            return value.to(device, dtype)
    if device is not None:
        return value.to(device)
    if dtype is not None:
        try:
            return value.to(dtype=dtype)
        except TypeError:
            return value.to(dtype)
    return value


def model_kwargs_from_inputs(inputs):
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return inputs


def decode_to_text(decoded):
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, (list, tuple)):
        return decode_to_text(decoded[0]) if decoded else ""
    return text_from_output(decoded)


def torch_dtype_from_compute_type(torch_module, compute_type, default=None):
    normalized = (compute_type or "").lower().replace("-", "_")
    if normalized in ("float16", "fp16", "half"):
        return getattr(torch_module, "float16", default)
    if normalized in ("bfloat16", "bf16"):
        return getattr(torch_module, "bfloat16", default)
    if normalized in ("float32", "fp32", "full", "default"):
        return getattr(torch_module, "float32", default) if normalized != "default" else default
    return default
