"""Native thinking transport for one-call gameplay; final action ABI is unchanged."""
from .interface import ResponseError, parse_response


def parse_thinking_action(text, prompt, *, current_version, thinking):
    if text == '__CONTEXT_OVERFLOW__':
        raise ResponseError('context_overflow')
    if thinking:
        # Qwen's chat template supplies the opening <think> before generation.
        # Never extract JSON from an unfinished reasoning segment.
        if '</think>' not in text:
            raise ResponseError('unfinished_thinking')
        if text.count('</think>') != 1:
            raise ResponseError('invalid_thinking_delimiters')
        _, text = text.split('</think>', 1)
    return parse_response(text.strip(), prompt, current_version=current_version)
