"""Coverage and accounting checks for the cloud shard collector."""
import json
from pathlib import Path
import tempfile
import unittest

from merge_thinking_shards import merge


class ShardMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.jobs=[dict(id=f'm{seed}s{seat}',seed=seed,seat=seat,board_hash=str(seed),
                        opponent='balanced',rng_index=2*(seed-1)+seat-1)
                   for seed in [1,2] for seat in [1,2]]
        self.write(self.root/'protocol.json',dict(jobs=self.jobs,replicas_per_mode=2))
        for shard in [0,1]:
            base=self.root/'shards'/str(shard)
            jobs=self.jobs[2*shard:2*shard+2]
            self.write(base/'protocol.json',dict(jobs=jobs))
            for mode in ['off','on']:
                out=base/mode
                games=[dict(j,status='native',reward=-1) for j in jobs]
                self.write(out/'runtime.json',dict(revision='pinned',thinking=mode=='on'))
                self.write(out/'games/games.json',games)
                self.write(out/'games/progress.json',dict(active=[],generating=[],elapsed_seconds=10+shard))
                self.write(out/'report.json',dict(games=games,complete=True,elapsed_seconds=10+shard))
                self.write(base/(mode+'-execution.json'),dict(exit_code=0,worker_seconds=20+shard))
                (out/'generation-timing.jsonl').write_text(''.join(json.dumps(dict(id=j['id'],decision=1))+'\n' for j in jobs))
                for j in jobs:(out/'games'/(j['id']+'.jsonl.gz')).write_bytes(b'fixture')

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_final_preserves_disjoint_coverage_and_costs(self):
        merge(self.root,final=True)
        for mode in ['on','off']:
            report=json.loads((self.root/mode/'report.json').read_text())
            self.assertEqual({x['id'] for x in report['games']},{j['id'] for j in self.jobs})
            self.assertEqual(report['elapsed_seconds'],11)
            self.assertEqual(report['gpu_worker_seconds'],41)
            self.assertEqual(len(list((self.root/mode/'games').glob('*.gz'))),4)

    def test_duplicate_shard_is_rejected(self):
        self.write(self.root/'shards/1/protocol.json',dict(jobs=self.jobs[:2]))
        with self.assertRaisesRegex(ValueError,'overlapping'):merge(self.root)

    def test_runtime_mismatch_is_rejected(self):
        self.write(self.root/'shards/1/off/runtime.json',dict(revision='different',thinking=False))
        with self.assertRaisesRegex(ValueError,'runtime mismatch'):merge(self.root)

    def test_missing_result_cannot_look_complete(self):
        path=self.root/'shards/1/off/report.json'
        row=json.loads(path.read_text());row['games']=row['games'][:1];self.write(path,row)
        with self.assertRaisesRegex(ValueError,'incomplete shard'):merge(self.root,final=True)

    def test_native_identity_change_is_rejected(self):
        path=self.root/'shards/0/off/games/games.json'
        rows=json.loads(path.read_text());rows[0]['seed']=999;self.write(path,rows)
        with self.assertRaisesRegex(ValueError,'identity changed'):merge(self.root)


if __name__=='__main__':unittest.main()
