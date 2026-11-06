"""Bind SFT rendering to completed prompt selection and immutable interfaces."""
import hashlib
import json
from pathlib import Path
from .interface import digest

ROOT=Path(__file__).resolve().parents[3]
PROMPT_SOURCES=('interface','encoding','memory','adapter','prompt_search','data','freeze')


def inference_config(config):
    return {key:config[key] for key in ('model_id','revision','max_length','max_new_tokens')}


def verify_prompt(manifest, config):
    if not manifest or manifest.get('schema')!='frozen-sft-prompt-v1':
        raise ValueError('real training requires a frozen prompt manifest')
    payload={k:v for k,v in manifest.items() if k!='manifest_hash'}
    if digest(payload)!=manifest.get('manifest_hash') or manifest['inference_config_hash']!=digest(inference_config(config)):
        raise ValueError('frozen prompt/config digest mismatch')
    from .prompt_search import STRATEGIES
    if STRATEGIES.get(manifest['candidate'])!=manifest['strategy']:
        raise ValueError('frozen strategy changed')
    for name,sha in manifest['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('frozen prompt interface source changed: '+name)
    for receipt in manifest['evidence']:
        path=ROOT/receipt['path']
        if path.parent.joinpath('invalidation.json').exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
            raise ValueError('frozen prompt evidence changed or invalidated')
    return manifest['strategy']


def freeze(output, *, search_games, screen_report, qualification, confirmation_games=None):
    from .prompt_search import STRATEGIES
    from .selection import select_prompt
    from .training import load_tokenizer
    evidence=[Path(p) for p in (search_games,screen_report,qualification)]
    screen=json.loads((ROOT/screen_report).read_text())
    if not screen.get('proceed_to_gameplay') or not json.loads((ROOT/qualification).read_text()).get('teacher_qualified'):
        raise ValueError('teacher qualification and screen must pass')
    search=json.loads((ROOT/search_games).read_text())
    if not search['complete']:raise ValueError('prompt search games incomplete')
    names=['original',*screen['selected_for_gameplay']]
    config=json.loads((ROOT/'configs/sft.json').read_text())
    splits=json.loads((ROOT/'runs/sft0/splits-v1/maps.json').read_text())
    evidence.append(Path('runs/sft0/splits-v1/maps.json'))
    def validate_games(path, report, candidates, split, opponent):
        expected={(name,m['seed'],m['board_hash'],seat,opponent,STRATEGIES[name])
                  for name in candidates for m in splits['maps'][split] for seat in (1,2)}
        actual={(g['candidate'],g['seed'],g['board_hash'],g['seat'],g['opponent'],g['strategy']) for g in report['games']}
        if actual!=expected or len(actual)!=len(report['games']):
            raise ValueError('gameplay differs from reserved candidate/map/seat/opponent allocation')
        model_path=Path(path).parent.parent/'model.json'
        allocation_path=Path(path).parents[2]/'protocol.json'
        model=json.loads((ROOT/model_path).read_text())
        allocation=json.loads((ROOT/allocation_path).read_text())
        if model['config']!=config or allocation['config']!=config:
            raise ValueError('model/config differs from scored gameplay')
        for name,sha in allocation['sources'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('source changed since scored gameplay: '+name)
        evidence.extend([model_path,allocation_path])
        return model
    model=validate_games(search_games,search,names,'prompt_search','weak')
    result=select_prompt(search['games'],names)
    selected=result['selected'];confirmation=None
    if selected!='original':
        if not confirmation_games:raise ValueError('revised prompt requires reserved confirmation games')
        evidence.append(Path(confirmation_games))
        confirm=json.loads((ROOT/confirmation_games).read_text())
        if not confirm['complete']:raise ValueError('confirmation games incomplete')
        validate_games(confirmation_games,confirm,['original',selected],'prompt_confirmation','pressure')
        confirmation=select_prompt(confirm['games'],['original',selected])
        selected=confirmation['selected']
    tokenizer=load_tokenizer(config)
    if model['tokenizer_hash']!=digest(tokenizer.backend_tokenizer.to_str()) or model['chat_template_hash']!=digest(tokenizer.chat_template):
        raise ValueError('tokenizer differs from scored gameplay')
    sources=[ROOT/f'src/polytopia_bot/learning/{name}.py' for name in PROMPT_SOURCES]
    sources.append(ROOT/'src/polytopia_bot/gpu/rules.json')
    manifest=dict(schema='frozen-sft-prompt-v1',candidate=selected,strategy=STRATEGIES[selected],
        inference_config_hash=digest(inference_config(config)),tokenizer_hash=digest(tokenizer.backend_tokenizer.to_str()),
        chat_template_hash=digest(tokenizer.chat_template),search_selection=result,confirmation=confirmation,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        evidence=[dict(path=str(p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in evidence])
    manifest['manifest_hash']=digest(manifest)
    verify_prompt(manifest,config)
    with (ROOT/output).open('x') as stream:json.dump(manifest,stream,indent=2)
    return manifest


def retain_original(output, *, screen_report, qualification):
    """Explicit original-prompt fallback when the fixed screen has no pair.

    This does not mark the failed comprehension gate as passed and does not
    invent a gameplay result. The unchanged original remains the base/SFT
    interface; subsequent smoke and final model gates still apply.
    """
    import gzip
    from .training import load_tokenizer
    from .prompt_search import STRATEGIES
    screen_path=Path(screen_report);qualification_path=Path(qualification)
    screen=json.loads((ROOT/screen_path).read_text())
    if screen.get('proceed_to_gameplay') or len(screen['selected_for_gameplay'])>=2:
        raise ValueError('screen has a viable pair; gameplay selection is required')
    if not json.loads((ROOT/qualification_path).read_text()).get('teacher_qualified'):
        raise ValueError('teacher qualification required')
    results_path=screen_path.parent/'results.json.gz'
    protocol_path=screen_path.parent/'protocol.json'
    with gzip.open(ROOT/results_path,'rt') as stream:results=json.load(stream)
    protocol=json.loads((ROOT/protocol_path).read_text())
    summaries=[r['summary'] for r in results['results']]
    if len(summaries)!=7 or {r['candidate'] for r in summaries}!=set(STRATEGIES) or any(r['decisions']!=100 for r in summaries):
        raise ValueError('the complete fixed 700-position screen is required')
    if summaries!=screen['summaries'] or protocol['strategies']!=STRATEGIES:
        raise ValueError('screen results or candidate texts changed')
    original=next(r for r in summaries if r['candidate']=='original')
    if original['first_attempt_valid']<protocol['first_attempt_valid_min'] or original['eventually_valid']<protocol['eventually_valid_min']:
        raise ValueError('original prompt fails the existing format/legality thresholds; diagnose before freezing')
    config=json.loads((ROOT/'configs/sft.json').read_text())
    if config!=results['config']:raise ValueError('config changed since screen')
    tokenizer=load_tokenizer(config)
    if digest(tokenizer.backend_tokenizer.to_str())!=results['tokenizer_hash'] or digest(tokenizer.chat_template)!=results['chat_template_hash']:
        raise ValueError('tokenizer changed since screen')
    qualified=json.loads((ROOT/qualification_path).parent.joinpath('protocol.json').read_text())['source_sha256']
    for name in ('interface','encoding','memory','adapter','teacher','prompt_search'):
        path=f'src/polytopia_bot/learning/{name}.py'
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=qualified[path]:
            raise ValueError('public interface or teacher changed since qualification')
    sources=[ROOT/f'src/polytopia_bot/learning/{name}.py' for name in PROMPT_SOURCES]
    sources.append(ROOT/'src/polytopia_bot/gpu/rules.json')
    evidence=[screen_path,protocol_path,results_path,qualification_path]
    manifest=dict(schema='frozen-sft-prompt-v1',candidate='original',strategy='',
        inference_config_hash=digest(inference_config(config)),tokenizer_hash=results['tokenizer_hash'],
        chat_template_hash=results['chat_template_hash'],
        search_selection=dict(selected='original',screen_passed=False,gameplay_selection='not_run_no_eligible_pair',
            reason='Retain the unchanged original when the bounded search produces no eligible pair; do not expand the search or claim prompt improvement.',summaries=summaries),
        confirmation=None,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        evidence=[dict(path=str(p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in evidence])
    manifest['manifest_hash']=digest(manifest);verify_prompt(manifest,config)
    with (ROOT/output).open('x') as stream:json.dump(manifest,stream,indent=2)
    return manifest
