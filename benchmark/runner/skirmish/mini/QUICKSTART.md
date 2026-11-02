# Start here

1. `/bot/src/main.rs` is the only submitted source. It already implements persistent
   line-based stdin/stdout, JSON parsing, flushing, and a placeholder action choice.
   Replace its decision logic. Files persist between coding stages; shell variables do not.
2. `/reference/PROTOCOL.md` specifies observation/action fields, IDs, and limits.
   Action IDs are current-position strings. Select an ID from `legal_actions`.
3. `/reference/RULES.md` is the complete searchable rule book and engine source.
   Its contents and heading index identify mechanics and data tables. Search it with
   `rg` or `grep`; it need not be printed into the conversation in one command.
4. `skirmish compile` builds the current source offline. `skirmish check` tests
   persistent WASI transport. `skirmish maps`, `inspect`, `replay`, and `example`
   expose the public test mechanics described in `/reference/TOOLS.md`.
5. No opponent or opponent code is supplied. Self-written experiments consume coding time.
   In the revision track, the controller supplies six expert games after each of
   stages 0, 1 and 2. Start with `/feedback/round-N/report.json`: results, public
   action counts, and own-empire observations by round. Full traces remain available.
6. The controller kills coder processes at the deadline and captures the exact current
   source, then independently compiles it. Current broken code fails; earlier versions
   are not substituted. Preserve a working final file before the deadline.

This guide describes interfaces, not a competitive strategy or reference opponent.
