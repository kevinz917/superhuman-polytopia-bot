"""Host-side inference routing. Local OpenAI/Anthropic defaults use coding plans.

Explicit OpenRouter configs remain OpenRouter; a failed proxy never falls back to
a metered API. Only the proxy's client key is read, never its OAuth credentials.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

PROXY_BASE = 'http://127.0.0.1:8317/v1'


def model_family(model):
    if model.startswith(('openai/', 'gpt-', 'codex-')):
        return 'openai'
    if model.startswith(('anthropic/', 'claude-')):
        return 'anthropic'
    raise ValueError('Specify provider explicitly for a model outside OpenAI/Anthropic')


def proxy_model(model):
    """Translate vendor prefixes/version punctuation, never substitute a model."""
    for prefix in ('openai/', 'anthropic/'):
        model = model.removeprefix(prefix)
    return re.sub(r'^(claude-(?:opus|sonnet|haiku|fable)-\d+)\.(\d+)', r'\1-\2', model)


@dataclass(frozen=True)
class Endpoint:
    backend: str
    base_url: str
    model: str
    key_env: str | None = None

    def credential(self):
        if self.key_env:
            key = os.environ.get(self.key_env, '')
        elif self.backend == 'cliproxyapi':
            key = os.environ.get('CLIPROXY_API_KEY', '')
            if not key:
                path = Path(os.environ.get('CLIPROXY_KEY_FILE',
                            '~/.cli-proxy-api/client-keys.json')).expanduser()
                try:
                    key = json.loads(path.read_text())['api_key']
                except FileNotFoundError:
                    raise ValueError('CLIProxyAPI client key missing; set CLIPROXY_API_KEY or CLIPROXY_KEY_FILE') from None
                except (ValueError, KeyError, TypeError):
                    raise ValueError('Invalid CLIProxyAPI client key file') from None
        else:
            key = ''
        if not isinstance(key, str) or not key.strip():
            raise ValueError('Missing inference credential for ' + self.backend)
        return key

    def receipt(self):
        # Safe to persist: no key, key file, or OAuth data.
        return dict(backend=self.backend, base_url=self.base_url, model=self.model,
                    accounting='subscription_usage' if self.backend == 'cliproxyapi' else 'provider_usage')


def endpoint(config, *, protocol='chat'):
    provider = config.get('provider', 'auto')
    if provider not in ('auto', 'openai', 'anthropic', 'chat', 'responses', 'cliproxyapi', 'openrouter'):
        raise ValueError('Unsupported provider adapter')
    transport = config.get('transport', 'auto')
    if transport not in ('auto', 'cliproxyapi', 'direct'):
        raise ValueError('Unsupported inference transport')
    base = config.get('base_url')
    if provider == 'openrouter':
        if transport == 'cliproxyapi':
            raise ValueError('Use provider=cliproxyapi, not openrouter, for the coding-plan proxy')
        backend = 'openrouter'
        base = base or 'https://openrouter.ai/api/v1'
        key_env = config.get('key_env', 'OPENROUTER_API_KEY')
    else:
        if provider == 'auto' and not base:
            model_family(config['model'])
        use_proxy = provider == 'cliproxyapi' or transport == 'cliproxyapi' or (transport == 'auto' and not base)
        if use_proxy:
            backend = 'cliproxyapi'
            base = base or os.environ.get('CLIPROXY_BASE_URL') or PROXY_BASE
            key_env = config.get('key_env')
        else:
            backend = 'direct'
            base = base or ('https://api.anthropic.com/v1' if protocol == 'anthropic' else 'https://api.openai.com/v1')
            key_env = config.get('key_env', 'ANTHROPIC_API_KEY' if protocol == 'anthropic' else 'OPENAI_API_KEY')
    base = base.rstrip('/')
    parsed = urlsplit(base)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or
            parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Inference base URL must be HTTP(S), without credentials, query or fragment')
    if backend == 'cliproxyapi' and os.environ.get('MODAL_TASK_ID') and parsed.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Local CLIProxyAPI is unreachable from Modal; configure an explicitly reachable proxy endpoint')
    model = proxy_model(config['model']) if backend == 'cliproxyapi' else config['model']
    return Endpoint(backend, base, model, key_env)


def parameters(config, route, *, protocol='chat'):
    """Convert OpenRouter-only options when using the compatible local proxy."""
    values = dict(config.get('parameters', {}))
    if route.backend == 'cliproxyapi':
        for name in ('provider', 'usage', 'plugins', 'models'):
            values.pop(name, None)
        if protocol == 'chat':
            reasoning = values.pop('reasoning', None)
            if reasoning and reasoning.get('effort'):
                values.setdefault('reasoning_effort', reasoning['effort'])
        elif protocol == 'responses' and 'max_tokens' in values:
            values.setdefault('max_output_tokens', values.pop('max_tokens'))
    return values
