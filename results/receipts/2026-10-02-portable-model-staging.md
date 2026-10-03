# Portable model staging receipt

Date: 2026-10-02 UTC

## Outcome

Both Sparks now have the exact portable checkpoint used by the Mac:
`Vontra/GLM-5.3-Flash-MLX-4bit-MTP` at revision
`76add2a341a1cd90ad0e86bb69839ea9c35827c6`.

Each verified copy contains 54 files and 181,741,759,037 bytes. Both match the
Mac content SHA-256
`b1b218bca5c72e28d77f5f4584b3bf12cacef97ba6df59f45a0094a3a041af07`
and filename-plus-size SHA-256
`43c3d20fab8f95b719856492c45b597905f8dba3f4eca24ea8466dae5c38b0c3`.

The successful `MODEL-STAGE-002` raw log is
`results/raw/20261002T221932Z-MODEL-STAGE-002.log`; its SHA-256 is
`69ce1881af9f37a58f59c02cb3abbc5bb2530bf597921d2375af5a5d2020d1d7`.

## Preserved failure and repair

The first complete hash gate stopped promotion. Shards 9 through 15 had been
resumed as nonempty but truncated files, and the checkpoint's PNG asset was
not selected. The failed assembly contained 53 files and 175,570,843,219
bytes, with content SHA-256
`be73dc60b7f4aa01cfc9496c05f1e040a4f9a91d03a9701794049d39bc40ca08`.

Only the seven invalid file tails and the missing small asset were copied from
the verified Mac checkpoint. Those repairs were then propagated over the
private Spark link, applied to both clean staging trees, and followed by a
second full content hash on both machines. The old incomplete head snapshot
was preserved under a `.pre-verified-20261002T222333Z` suffix.

The downloader now requests file metadata, checks the exact remote byte size
of every selected file, force-downloads mismatches, verifies the repair, and
includes PNG assets. Its automated regression tests cover both selection and
truncated-file recovery.

## Remaining gate

This receipt proves model identity only. Performance is not accepted until the
same-model two-Spark reference and full Spark-prefill/MCDMA/Mac-decode path run
the same fixed workload and the latter clears the project's 3% median gain
threshold with matching output.
