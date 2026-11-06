"""Native Anthropic wire format around the unchanged Mini-SWE bash parser/loop.

Signed thinking and redacted thinking blocks are round-tripped byte-for-byte as
JSON values, rather than reconstructed from an OpenAI compatibility translation.
"""
import copy
import json


def request_payload(messages, options, model, bash_tool):
    system = []
    turns = []
    for message in messages:
        role = message['role']
        content = message.get('content') or ''
        if role == 'system':
            system.append(content)
            continue
        if role == 'tool':
            role = 'user'
            blocks = [dict(type='tool_result',tool_use_id=message['tool_call_id'],content=content)]
        elif role == 'assistant' and '_anthropic_content' in message:
            blocks = copy.deepcopy(message['_anthropic_content'])
        else:
            blocks = [dict(type='text',text=content)] if isinstance(content,str) else copy.deepcopy(content)
            if role == 'assistant':
                for call in message.get('tool_calls') or []:
                    blocks.append(dict(type='tool_use',id=call['id'],name=call['function']['name'],
                                       input=json.loads(call['function']['arguments'])))
        if turns and turns[-1]['role'] == role:
            turns[-1]['content'].extend(blocks)
        else:
            turns.append(dict(role=role,content=blocks))
    values = copy.deepcopy(options)
    effort = values.pop('reasoning_effort',None)
    reasoning = values.pop('reasoning',None)
    if not effort and reasoning:
        effort = reasoning.get('effort')
    for name in ['usage','provider','plugins','models']:
        values.pop(name,None)
    if effort:
        values['thinking'] = dict(type='adaptive')
        values['output_config'] = dict(effort=effort)
    tool = bash_tool['function']
    return dict(values,model=model,system='\n\n'.join(system),messages=turns,
        tools=[dict(name=tool['name'],description=tool['description'],input_schema=tool['parameters'])])


def response_envelope(raw):
    blocks = raw['content']
    message = dict(role='assistant',content=''.join(b.get('text','') for b in blocks if b['type']=='text'),
                   _anthropic_content=copy.deepcopy(blocks))
    calls = [dict(id=b['id'],type='function',function=dict(name=b['name'],arguments=json.dumps(b['input'])))
             for b in blocks if b['type']=='tool_use']
    if calls:
        message['tool_calls'] = calls
    original = raw.get('usage') or {}
    usage = dict(prompt_tokens=sum(original.get(k,0) for k in
        ['input_tokens','cache_creation_input_tokens','cache_read_input_tokens']),
        completion_tokens=original.get('output_tokens',0),anthropic_usage=original)
    if 'cost' in original:
        usage['cost'] = original['cost']
    finish = {'tool_use':'tool_calls','max_tokens':'length','refusal':'content_filter'}.get(raw.get('stop_reason'),'stop')
    return dict(id=raw.get('id'),model=raw.get('model'),choices=[dict(index=0,finish_reason=finish,message=message)],
                usage=usage,provider_response=raw)
