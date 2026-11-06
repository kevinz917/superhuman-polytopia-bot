"""Compress prepared tokens before dispatch without changing their audited bytes."""
import gzip,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from polytopia_bot.learning.training import file_hash


def main(root):
 root=Path(root)
 if (root/'call.json').exists():raise ValueError('cannot change dispatched protocol')
 p=json.loads((root/'protocol.json').read_text());raw='prepared/tokens.jsonl';packed=raw+'.gz'
 if file_hash(root/raw)!=p['uploads'][raw]:raise ValueError('prepared tokens changed')
 with (root/raw).open('rb') as source,gzip.open(root/packed,'wb',compresslevel=1) as out:shutil.copyfileobj(source,out)
 p['compressed_inputs']={packed:raw};p['upload_files']=[packed if name==raw else name for name in p['upload_files']]
 p['uploads'][packed]=file_hash(root/packed)
 (root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps({'original_bytes':(root/raw).stat().st_size,'uploaded_bytes':(root/packed).stat().st_size}))

if __name__=='__main__':main(sys.argv[1])
