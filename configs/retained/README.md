# `configs/retained/`

Phases 3–6 build on the winning run of the previous phase instead of
`base.yaml` directly. Once a phase's results are in, materialize the
choice as a file here named after the phase, e.g. `phase2.yaml`:

```yaml
inherit: base
regime: bptt   # whichever of p2_pushforward / p2_bptt won
```

Then a phase-3 run config says `inherit: phase2` instead of `inherit: base`,
and `load_config()` resolves it from this directory automatically (see
`_resolve_inherit` in `src/beamsurrogate/config.py`).

Not yet populated: phases 0–2 (`configs/runs/*.yaml`, all inheriting
`base` directly) haven't been run yet, so there is no winner to record.
