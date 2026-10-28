"""Install the local asset cache of Polytopia graphics for the Studio (personal use; not redistributed)."""
import concurrent.futures, hashlib, json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads(Path(__file__).with_name('mobile-assets.json').read_text())
dest=ROOT/'artifacts/mobile-assets';dest.mkdir(parents=True,exist_ok=True)
def install(entry):
    path=dest/entry['file']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:return
    with urlopen(entry['source'],timeout=60) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Publisher asset changed: '+entry['file'])
    path.write_bytes(raw)
with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(install,manifest.values()))
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Verified',len(manifest),'local graphics. Midjiwan retains ownership; do not redistribute the asset cache.')
