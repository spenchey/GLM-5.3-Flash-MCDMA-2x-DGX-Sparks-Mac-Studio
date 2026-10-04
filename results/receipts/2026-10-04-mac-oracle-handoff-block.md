# Mac oracle opening-block upper bound

Experiment: `MAC-ORACLE-BLOCK-001`

## Question

Could the Sparks draft an opening reply while MCDMA transfers the prompt cache,
then let the Mac verify that opening cheaply enough to clear the frozen
6.572681-second complete-time target?

## Qualification

This was an intentionally optimistic upper bound. It supplied the Mac with the
known-correct greedy tokens and excluded Spark draft production, MCDMA transfer,
and first-token verification. A deployable implementation can only be slower.

## Result

| Opening block | Exact runs | Median verify | Median verify + remainder |
|---:|---:|---:|---:|
| 3 | 3/3 | 0.037306 s | 6.713734 s |
| 8 | 3/3 | 0.084360 s | 6.642427 s |
| 16 | 3/3 | 0.159009 s | 6.582038 s |
| 24 | 0/3 | 0.156091 s | invalid |
| 32 | 0/3 | 0.184905 s | invalid |
| 48 | 0/3 | 0.279214 s | invalid |
| 64 | 0/3 | 0.340438 s | invalid |

The widest exact opening was 16 tokens. Even its optimistic 6.582038-second
median misses the complete-path target before any transfer is counted. Wider
passes deterministically changed the later output at tokens 90, 104, 104, and
181 respectively.

Reference decode: 6.893602 seconds, 60.540675 decode tokens/s. All valid runs
kept token SHA-256
`9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`.

## Evidence

- Harness: `scripts/benchmark-mac-oracle-handoff-block.py`
- Raw log: `results/raw/20261004T025113Z-MAC-ORACLE-BLOCK-001.log`
- Raw-log SHA-256:
  `40174743ce0e216a41759ae9680bd081a2101660eee5f1e40d9777f005ef9f17`

## Decision

Do not build the distributed opening-block path. The exact window cannot pay
the handoff, and wider MLX prompt-kernel state is not continuation-exact.
