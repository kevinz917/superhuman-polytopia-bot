import json, sys, os
for p in sys.argv[1:]:
    j = p[:-4] + '.json'
    if not os.path.exists(j): print(os.path.basename(os.path.dirname(p)) or p, 'NO JSON'); continue
    d = json.load(open(j)); c = d.get('config', {})
    print(p.replace('/vol/results/distributed/','').replace('/vol/retained/checkpoints/distributed-final-v1/',''), '|', d.get('stage'), '| params', d.get('parameter_elements'),
          '|', {k: c.get(k) for k in ('architecture','width','blocks','extra_blocks','global_pool','block_pool','obs_version','attention','players','size','history')})
