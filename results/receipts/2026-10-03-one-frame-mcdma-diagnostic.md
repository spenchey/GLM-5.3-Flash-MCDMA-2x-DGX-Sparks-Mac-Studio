# One-frame MCDMA cache-handoff diagnostic

The deployed recipe commit was `9a1974d`. TensorFold remained pinned to
`609ca419abecebdc5a059498a613680bd3aa847f` (v0.6.5), MCDMA to `7192192`, and
the portable GLM checkpoint to
`76add2a341a1cd90ad0e86bb69839ea9c35827c6`. The only intentional runtime
change from MIA-DFLASH-001 was increasing both the MCDMA reply mailbox and the
application frame to 160 MiB, large enough for the 148,032,512-byte cache.
The later decoder audit reclassified both runs as MTP: the one-frame transport
timings remain valid, but neither run is DFlash2 performance evidence.

The maximum 256 MiB mailbox was rejected before model startup because the Mac
failed to register mailbox segment 64. A 160 MiB mailbox started correctly and
the cache crossed as one frame. Its first-ever call paid a one-time 2.12-second
handoff and 1.18-second import cost. After that initialization, the measured
request reported:

- one 148,032,512-byte frame;
- 0.508989 seconds inside the handoff and 0.509739 seconds observed transfer;
- 0.006212 seconds to import;
- 7.004234 seconds complete;
- 61.511 decode tokens/s;
- the same stable candidate token SHA-256 as MIA-DFLASH-001;
- the same first Mac-control difference at token 104.

The prior three-frame run's median handoff was 0.517833 seconds. One framing
call therefore saved only 0.008844 seconds, leaving the result 0.431553 seconds
above the 6.572681-second acceptance ceiling. One frame is valid and slightly
faster after warmup, but it cannot close the goal by itself.

Raw evidence:

- first-use result: `results/raw/mia-c1-dflash-one-frame-20261003T231752Z-73334/result.json`,
  SHA-256 `c2f906ed4aebe6dceacd8d558e21ea1a89f7c091d057bcbcb2cbf4fd2c1edb54`;
- initialized result: `results/raw/mia-c1-dflash-one-frame-warm-20261003T232013Z-74025/result.json`,
  SHA-256 `b0f74da5f8af8d3198b14fa79edc2330523c7811e1f41a5bbeedc68938a3bebb`;
- clean-stop evidence: `results/raw/stopped-cache-handoff-20261003T232256Z-74948/`.

No service remains running.
