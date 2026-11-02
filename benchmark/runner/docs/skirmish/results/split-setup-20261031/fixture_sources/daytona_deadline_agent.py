import asyncio,json
from pathlib import Path
from polytopia_bot.skirmish.harbor_agent import SkirmishAgent
from polytopia_bot.skirmish.local import run_stage
from polytopia_bot.skirmish.common import sha,write_json

class DeadlineAgent(SkirmishAgent):
    async def run(self,instruction,environment,context):
        try:
            original=await asyncio.to_thread(self.sandbox.filesystem.read_bytes,'/bot/src/main.rs')
            payload=dict(self.config,seconds=8,task='Run the supplied long command.',system='Use bash.',trajectory=str(self.logs_dir/'trajectory.json'))
            reason,timing=await asyncio.to_thread(run_stage,self.sandbox,self.logs_dir,0,payload,None,6)
            capture=json.loads(await asyncio.to_thread(self.sandbox.filesystem.read_text,'/verify/cutoffs/stage-0/capture.json'))
            marker=await asyncio.to_thread(self.sandbox.filesystem.read_text,'/bot/deadline-command-started')
            await asyncio.sleep(32)
            after=await asyncio.to_thread(self.sandbox.filesystem.read_bytes,'/bot/src/main.rs')
            assert marker.strip()=='started'
            trajectory=json.loads((self.logs_dir/'trajectory.json').read_text())
            exit_status=trajectory.get('info',{}).get('exit_status')
            assert (reason=='deadline' or exit_status=='TimeExceeded') and 0<timing['coding_seconds']<=8
            assert sha(original)==sha(after)==capture['source_sha256']
            report=dict(passed=True,fixture=True,actual_model_calls=0,budget_seconds=8,long_command_started=True,source_unchanged_after_late_write_window=True,capture=capture,timing=timing)
            write_json(self.logs_dir/'deadline-check.json',report)
            context.metadata=report
        finally:
            if self.bridge:
                await asyncio.to_thread(self.bridge.close)
                self.bridge=None
