# Local coding-plan inference

CLIProxyAPI is available on this machine at `http://127.0.0.1:8317/v1`. New local OpenAI and Anthropic calls default to it. This applies to Chat Completions, OpenAI Responses, Anthropic Messages, and the Mini-SWE-agent model adapter. Proxy failures never switch to a paid API or another model. Existing explicit `provider: openrouter` configurations retain their meaning and the completed benchmark is unchanged.

```python
from polytopia_bot.skirmish.providers import provider
from polytopia_bot.skirmish.mini_worker import coding_model

# Native Responses and Messages protocols, with proxy routing by default.
openai = provider({"model": "openai/gpt-6-luna"}, "System instruction", tools=[])
anthropic = provider({"model": "anthropic/claude-fable-5-1"}, "System instruction", tools=[])

# Same upstream Mini-SWE-agent parser, bash tools and continuation behavior.
mini = coding_model({"model": "anthropic/claude-fable-5-1",
                     "parameters": {"reasoning": {"effort": "high"}, "max_tokens": 32768}})
```

`provider: chat`, `responses`, `openai`, and `anthropic` select a wire protocol. `provider: cliproxyapi` explicitly selects proxy Chat Completions. Missing/`auto` provider infers OpenAI Responses or Anthropic Messages from the model name. Mini always uses Chat Completions with its upstream tool parser. Vendor prefixes are removed on proxy requests; Claude version punctuation is normalized (`claude-opus-5.5` → `claude-opus-5-5`). Unknown models are never replaced. Both `claude-fable-5` and `claude-fable-5-1` are supported; the local roster selects 5.1.

The client key comes from `CLIPROXY_API_KEY`, or the `api_key` field in `~/.cli-proxy-api/client-keys.json`. `CLIPROXY_KEY_FILE` and `CLIPROXY_BASE_URL` override those locations. No upstream OAuth tokens are read. Keys remain in the trusted process and the killable HTTP worker's stdin, never the coder sandbox, command arguments, source configs or trajectory serialization. Receipts record the backend, endpoint, exact wire model and subscription accounting mode; absent dollar costs do not stop a run and are not presented as an invoice of zero.

For an intentional metered call, use `provider: openrouter`, or `transport: direct` with an OpenAI/Anthropic provider. An explicit `base_url` also selects that endpoint. Direct credentials use `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`; OpenRouter uses `OPENROUTER_API_KEY`. `key_env` can explicitly name a different credential variable.

The local loopback address cannot be reached from a Modal controller. Automatic proxy routing rejects loopback there. A remotely hosted trusted controller needs an explicitly reachable authenticated proxy endpoint; this adapter does not expose the local service or silently tunnel it. The existing official Modal cohort launcher remains explicitly OpenRouter-only. A local trusted Mini controller can use this model adapter with its existing offline Modal bash environment. This is a repository adapter, not a machine-wide SDK monkey patch.

Check the roster without inference, or run small native/Mini tool-continuation checks:

```sh
python scripts/check_skirmish_proxy.py
python scripts/check_skirmish_proxy.py --live
```

The roster is in `configs/skirmish/cli-proxy-models.json`; it is adapter input, not an official OpenRouter cohort manifest. Adding Fable here does not assign it a benchmark score or launch a run. Research execution and raw benchmark artifacts remain on Modal.

## Full-cohort cost estimate — October 31, 2026

For the previous eleven models plus Fable 5.1, with three builds on each forty-minute track, plan for **roughly $120–160 incremental spend**, excluding existing coding-plan subscriptions. This is a forecast, not an invoice or a dollar gate. It assumes the four existing OpenAI/Anthropic models and Fable are covered by those plans, while the other seven models remain on OpenRouter.

The selected authoring receipts previously totaled $271.8896. Removing Astra, Sol, Luna and Opus leaves approximately **$56.96** in OpenRouter charges. Including the previous non-proxied models' excluded infrastructure attempts raises that to approximately **$61.3**, plus a few cents of probes. Fable adds no per-token API charge under the stated subscription assumption. Provider changes can affect latency, completed work, caching and cost, so this does not promise identical calls or scores.

The completed cohort recorded 134,300.7 coding-stage seconds, 4,942.7 feedback-game seconds, and 170,363.2 held-out/peer game seconds across workers. These are summed task runtimes, not elapsed campaign hours or a billing export. At [Modal's current list prices](https://modal.com/pricing), non-preemptible CPU is $0.00003942/core-second and memory is $0.00000667/GiB-second. Applying the current two-core/6-GiB controller and game-worker requests, and two-core/2-GiB coding sandboxes, gives about $49.64 for these measured portions of an eleven-model rerun. Scaling authoring to twelve models and scheduled evaluation from 16,830 to 18,576 outcomes gives about **$54.41 before unmeasured overhead**. Budget **$60–90 for Modal compute** to allow for setup, freezing, independent rebuilds, replay auditing, warmup, coordinators and retries. Storage, unusual overruns and usage above requested allocations are not reconciled here; available Modal credits are not subtracted.

The twelve-model design has 72 final bots, 180 stage snapshots, 16,200 primary outcomes and 2,376 peer outcomes. A cohort using only the five subscription models would have no metered model-API charge under the same assumption, but would still consume coding-plan quota and Modal resources. No new benchmark was launched to produce this estimate.
