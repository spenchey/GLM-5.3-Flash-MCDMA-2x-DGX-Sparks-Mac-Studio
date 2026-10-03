# Three-machine lifecycle and recovery receipt

Date: 2026-10-02

Result: **PASS — three clean stop/start/answer cycles completed after three
recorded packaging and status-reader failures were repaired.**

## Successful cycles

Each accepted cycle used orderly model shutdown through MCDMA, clean MCDMA
cleanup, a fresh start, both TensorFold CUDA ranks, the Mac Metal stage, and an
exact `Paris` answer.

| Cycle | Start raw SHA-256 | Answer raw SHA-256 |
| ---: | --- | --- |
| 1 | `2b53849eadf1325ad2df93043b9bae9e826c0cb57e9745ab8893684133a51270` | `c4db65dc2a7d830fca66d306f5739300d66f6983a5c61e4bf4a0443d3dcc907e` |
| 2 | `c205f56401f0700f766fa658569cf7e10279bdf577f6287df549083e4e399481` | `b3a4afcbe5e14bcdcfc44e5c2590ecbb4ef0ee74e02a6492811958c7fbe59b80` |
| 3 | `6daf102f71163eee56f4990a06dfada80610059cd8c909a5c2ae953ea0afce46` | `ed25fa18a5ee5413baada36679996077045d8571c71b174afadb154b9adbb0d1` |

The final clean stop raw SHA-256 was
`51893a496d37fd798873fc8a19be985010634b16d2eaf10be3dba165672b4463`.
The service was then started again for the accepted persistent deployment.

## Preserved failures

1. **RECOVERY-002:** the adapter required a clean private directory to be
   absent, although a prior clean run left the empty directory. It now safely
   reuses a verified empty private directory. Raw SHA-256
   `b378dec3d76e9475660be6755f2a5836e8e5e289407ae52a8aa7a74a035330e2`.
2. **RECOVERY-003:** a host checkpoint path was passed inside a container. Host
   and container paths are now separate pins checked by preflight. Raw SHA-256
   `fa4dfe8a080441f523550d7c229fce0d349cb0b785a66f58112c1e2c5515a532`.
3. **RECOVERY-010:** the control reader assumed one socket read contained a
   complete multi-line reply and saw only `VERSION`. It now reads through the
   reply terminator. One CUDA rank remained blocked in NCCL after this failed
   startup and ignored TERM/INT; the two exact experimental containers were
   force-stopped once. No other container, model, disk, or MCDMA process was
   harmed. Raw SHA-256
   `c8a86cdbb090d62476392d34acae6399b1239a6e85289b6715a682cd6ccba02d`.

The final startup loop tolerates a transient read while the daemon becomes
ready, but never treats an incomplete or unhealthy reply as ready.

## Shutdown behavior

Normal stop signals the Mac API. The API closes its listener, sends an orderly
shutdown over MCDMA, and both ranks exit code zero. Only then are the connector,
listener, and SSH tunnel removed. Cleanup rejects surviving children,
unexpected problems, or a forced kill.

Forced stop remains an explicit failed-start recovery for the named
experimental containers only; it is not accepted as a clean lifecycle pass.
