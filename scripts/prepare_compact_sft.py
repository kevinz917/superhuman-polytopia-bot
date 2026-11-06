"""Re-render the exact expert-sft-v1 examples with the compact prompt and tokenize them (run on Modal).

shard(files, selected, config, out): ``selected`` maps job id -> {decision: label} from the
v1 ``prepared/index.jsonl``. Each selected decision's public state is recovered from its
stored readable prompt (proved exact by re-rendering it byte for byte), rendered compactly,
round-tripped, and labelled with the same ``aN``. ``combine`` is reused unchanged from
``prepare_expert_sft``.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_expert_sft import combine  # noqa: E402,F401
from polytopia_bot.learning.compact_prompt import (compact_messages, parse_compact, public_from_readable,  # noqa: E402
                                                   readable_from_public, without_rules)
from polytopia_bot.learning.data import tokenize_example  # noqa: E402
from polytopia_bot.learning.expert_data import read_game, response_text  # noqa: E402
from polytopia_bot.learning.interface import canonical, digest  # noqa: E402


def convert(header, ex):
    """Compact messages for one stored readable example, with every equivalence check."""
    public = public_from_readable(ex['user'])
    old = readable_from_public(public)
    if old[0]['content'] != header['system'] or old[1]['content'] != ex['user']:
        raise ValueError('stored readable prompt is not reproduced from its public state')
    messages = compact_messages(public)
    parsed = parse_compact(messages[1]['content'])
    if canonical(parsed) != canonical(without_rules(public)):
        raise ValueError('compact prompt does not round-trip the public state')
    legal = {e['action_id']: e['command'] for e in public['legal_actions']}
    if {e['action_id']: e['command'] for e in parsed['legal_actions']} != legal or ex['label'] not in legal:
        raise ValueError('compact catalog differs')
    return messages, legal[ex['label']]


def shard(files, selected, config, out):
    from polytopia_bot.learning.training import load_tokenizer
    tokenizer = load_tokenizer(config)
    stats = Counter(); out = Path(out); tmp = out.with_suffix('.tmp')
    with open(tmp, 'w') as sink:
        for path in files:
            header, examples, result = read_game(path)
            job = header['job']['id']
            want = selected.get(job, {})
            stats['games'] += 1
            found = 0
            for ex in examples:
                if str(ex['decision']) not in want:
                    continue
                if want[str(ex['decision'])] != ex['label']:
                    raise ValueError(f'label differs from v1 index for {job}:{ex["decision"]}')
                found += 1
                messages, command = convert(header, ex)
                tok = tokenize_example(tokenizer, messages, response_text(ex['label']),
                                       max_length=config['max_length'], max_new_tokens=config['max_new_tokens'])
                sink.write(json.dumps(dict(**tok, id=f"{job}:{ex['decision']}", family=ex['family'], turn=ex['turn'],
                    key=hashlib.sha256((messages[0]['content']+'\0'+messages[1]['content']).encode()).hexdigest(),
                    label=ex['label'], command=command))+'\n')
                stats['sampled'] += 1
            if found != len(want):
                raise ValueError(f'{job}: found {found} of {len(want)} selected decisions')
    tmp.replace(out)
    return dict(stats=dict(stats), tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
                chat_template_hash=digest(tokenizer.chat_template), path=str(out))
