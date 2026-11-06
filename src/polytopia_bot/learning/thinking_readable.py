"""Named fields for the exact same sparse public observations and legal catalog."""
from copy import deepcopy
from dataclasses import replace
import json

from .decision_harness import FAMILIES
from .interface import canonical
from .thinking_debug import render_sparse_decision


def records(value, inverse=False):
    if isinstance(value, list):
        return [records(v, inverse) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {'columns', 'rows'} and not inverse:
        return dict(columns=value['columns'], records=[dict(zip(value['columns'], row, strict=True)) for row in value['rows']])
    if set(value) == {'columns', 'records'} and inverse:
        return dict(columns=value['columns'], rows=[[row[k] for k in value['columns']] for row in value['records']])
    return {k: records(v, inverse) for k, v in value.items()}


def readable_messages(messages):
    result = deepcopy(messages)
    body = json.loads(result[1]['content'])
    if body['encoding'] != 'polytopia-compact-sparse-terrain-v1':
        raise ValueError('requires sparse terrain input')
    body = records(body)
    body['encoding'] = 'polytopia-named-fields-v1'
    # Labels are a reversible presentation aid, never action ranking.
    body['action_names'] = {row['action_id']: FAMILIES[row['kind']] for row in body['legal_actions']['records']}
    result[0]['content'] = result[0]['content'].replace(
        'Entity tables contain columns and\nrows;', 'Entity tables contain columns and named records;').replace(
        'Rule definitions also\nuse column tables', 'Rule definitions also use named records').replace(
        'Each row is one legal choice.', 'Each named record is one legal choice.').replace(
        'Input encoding: polytopia-compact-sparse-terrain-v1.', 'Input encoding: polytopia-named-fields-v1.')
    result[0]['content'] = result[0]['content'].replace(
        ' For example, to select the\nrow ["a17",1,23,24,0], return exactly {"action_id":"a17"}.', '')
    result[0]['content'] += ('\nRead named fields literally. action_names decodes each command kind. '
        'A positive unit home value is the home city tile plus one; zero means no home. '
        'The unit tile field is its actual location. '
        'Use the supplied facts and rules; do not infer extra action prerequisites. '
        'Make a decision once you have enough information; do not repeatedly reopen '
        'the same comparison. A short rationale is sufficient for one game action.')
    result[1]['content'] = canonical(body)
    return result


def restore_sparse_body(body):
    result = records(body, inverse=True)
    result.pop('action_names')
    result['encoding'] = 'polytopia-compact-sparse-terrain-v1'
    return result


def render_readable_decision(obs):
    prompt = render_sparse_decision(obs)
    return replace(prompt, messages=readable_messages(prompt.messages))


def action_schema(messages):
    catalog = json.loads(messages[1]['content'])['legal_actions']
    ids = ([row['action_id'] for row in catalog['records']] if 'records' in catalog
           else [row[0] for row in catalog['rows']])
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
        raise ValueError('invalid action catalog')
    return dict(type='object', properties={'action_id': dict(type='string', enum=ids)},
                required=['action_id'], additionalProperties=False)
