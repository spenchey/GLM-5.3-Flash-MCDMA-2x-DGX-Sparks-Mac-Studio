# P06B1 SSH command-quoting repair receipt

Date: 2026-10-01

Linear issue: `MOT-3368`

Base commit: `12749d6b0d9e6249d5c211e85d2f2eadb8773ea5`

Result: **PASS — OFFLINE REPAIR ONLY; NO LIVE HOST OR SERVICE CHANGED**

## Trigger and exact reproduction

The first P06C no-service attempt stopped before a fault stage because the
adapter reported `mailbox helper compile emitted unexpected output`. Its
cleanup then falsely reported that an already absent tunnel socket survived.

An offline fake SSH executable reproduced OpenSSH's remote-command behavior by
joining the arguments after the host and letting a login shell parse the joined
string. Before the repair, the focused adapter suite reproduced the live
symptoms:

- `umask 077; ...` emitted the unexpected line `0022`;
- stdin reached the compiler, but the leaked `0022` made compilation appear to
  fail;
- `test ! -e <missing socket>` returned `1` instead of `0`;
- a failing command preserved exit `7`, but its diagnostic omitted stdout and
  exposed an unredacted token-shaped value.

The pre-fix focused result was 12 tests run with three failures and one error.

The first independent review then found two additional fail-open paths in the
initial repair:

1. `Session.cleanup()` joined each host's process, socket, and listener checks
   with semicolons. A remaining socket in the middle could return failure and
   then be hidden by the final listener check returning success.
2. Field-oriented diagnostic replacement did not cover natural-language,
   failed-token, JSON, or URL-credential forms. Exact probes including
   `password is ...`, `token failed: ...`, a JSON token field, and URL userinfo
   retained their sensitive values.

Deterministic review tests reproduced both defects before the follow-up repair:
the active Spark socket produced `clean=true` when the later listener check
succeeded, and all four diagnostic forms retained their test secrets.

A second review found that word-boundary matching still treated underscores as
part of a word. Neutral-value probes therefore leaked unchanged for
`access_token`, `refresh_token`, `auth_token`, `client_secret`,
`session_cookie`, and `password_hash`; camel-case, query-string, and JSON
variants had the same gap. These probes were added before changing the matcher
and failed on the reviewed code.

A final review found that the conservative substring matcher overcorrected and
hid harmless diagnostics containing `tokenization`, `passwordless`,
`secretary`, `cookiecutter`, `credentialed`, or `private keyboard`. Exact
ordinary-output probes reproduced all six false positives before the boundary
repair.

The next review found that six Pascal-case key names still retained a neutral
value: `ClientSecret`, `AccessToken`, `RefreshToken`, `AuthToken`,
`SessionCookie`, and `PasswordHash`. Exact probes reproduced all six leaks
before adding their bounded first-prefix capitalization variants.

The following review found additional bounded common token keys outside that
list, including API, OAuth, ID, CSRF, session, and client prefixes. Seventeen
neutral-value lower-case, Pascal-case, and acronym-prefix probes reproduced the
gap before their explicit variants were added.

## Root cause and repair

`Remote.shell()` passed `sh`, `-c`, and the raw script as separate SSH
arguments. OpenSSH joins remote-command arguments with spaces. The remote login
shell therefore saw `sh -c umask 077; ...`: the inner shell received only
`umask`, printed its current mask, and the outer shell ran the remaining
command. The same split reduced `test ! -e ...` to a bare failing `test`.

The repair quotes the complete script as one POSIX-shell payload before it is
given to SSH. `Remote.compile()` now uses that same path while preserving its C
source on stdin. Command failures retain the real exit code and include bounded,
whitespace-normalized stdout and stderr; common credential-shaped assignments
are redacted before they enter the exception.

The review repair runs all six process/socket/listener absence predicates as
separate commands with an exact condition name. A later success therefore
cannot overwrite an earlier failure. Any nonzero exit or unexpected stdout or
stderr fails cleanup, but cleanup still performs no deletion or forced kill.
Diagnostics now redact the whole stream whenever a password, token, secret,
authorization, Bearer, credential, cookie, or API/access/private-key marker is
present. Marker components use explicit ASCII-alphanumeric boundaries, so
underscores, hyphens, and dots remain key separators without matching longer
ordinary words. Compound key names require a complete `key` component, and
known camel- and Pascal-case credential keys are matched separately and
case-sensitively, including explicit common API, OAuth, ID, CSRF, session, and
client token-key forms. URL userinfo is removed separately, while ordinary
bounded output is preserved.

## Offline proof

The fake-SSH tests now prove:

- semicolons, spaces, a glob, a dollar sign, and a single quote remain inside
  one remote script;
- stdin compilation succeeds and creates an owner-only (`0700`) binary under
  `umask 077`;
- exit `7`, stdout, and stderr are preserved while `token=...` is redacted;
- a missing socket makes `test ! -e` succeed;
- an active UNIX socket makes `test ! -e` fail and remains present;
- each of the Mac and Spark process, socket, and loopback-listener predicates
  independently makes `clean=false`, regardless of later predicate success;
- an active Spark socket produces the exact `Spark RPC socket remains` problem
  even when the subsequent listener predicate succeeds;
- password, failed-token, JSON token, Bearer, API-key, API_KEY, and URL-userinfo
  probes do not retain test credentials, while ordinary diagnostic text remains
  readable;
- neutral-value probes for `access_token`, `refresh_token`, `auth_token`,
  `client_secret`, `session_cookie`, `password_hash`, camel-case keys, query
  parameters, headers, and JSON all return only the redaction marker;
- the six exact Pascal-case credential-key probes also return only the
  redaction marker;
- lower-case, Pascal-case, and acronym-prefix token fields for API, OAuth, ID,
  CSRF, session, and client also return only the redaction marker;
- the six exact harmless diagnostics containing `tokenization`, `passwordless`,
  `secretary`, `cookiecutter`, `credentialed`, and `private keyboard` remain
  unchanged;
- bounded near-matches such as `apiTokenizer`, `OAuthTokenizer`, and
  `sessionTokenization` remain unchanged;
- the original status, cleanup-order, no-forced-kill, mailbox, and state tests
  remain green.

Validation commands and results:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/tests/p06-fault-helper.py scripts/tests/p06-live-adapter.py \
  tests/test-p06-fault-helper.py tests/test-p06-live-adapter.py
# PASS

PYTHONDONTWRITEBYTECODE=1 python3 tests/test-p06-fault-helper.py
# 12 tests passed

PYTHONDONTWRITEBYTECODE=1 python3 tests/test-p06-live-adapter.py
# 15 tests passed, including six independent cleanup-predicate subtests

bash -n scripts/mcdma-control-tunnel.sh tests/test-mcdma-control-tunnel.sh
# PASS; syntax only, so no SSH was opened

cc -std=c11 -O2 -Wall -Wextra -Werror -fsyntax-only \
  scripts/tests/mcdma-rpc-mailbox.c
# PASS

./tests/test-config.sh
# PASS: pinned inputs and private-data guard

./tests/test-restore-two-spark.sh
# PASS: local fake-SSH rollback contract; no real SSH was opened

git diff --check
# PASS
```

Final SHA-256 values:

- adapter: `b2bb86945b7c5401b3adb6f1693d54a1753a19e25260fe2695e226e66539ac66`
- adapter tests: `f40f169d76b775d58ffd3e5c7e77b5976b7cc2339555aa7060b51af9b5b12958`

## Scope and remaining gate

Only these files are owned by this repair:

- `scripts/tests/p06-live-adapter.py`
- `tests/test-p06-live-adapter.py`
- this receipt

No SSH session, MCDMA daemon, RDMA transfer, installed binary, GLM container,
network setting, or cable was touched. The proof models OpenSSH locally; it
does not claim that a live P06C fault stage has passed. P06C must start again
from its preflight after this repair and the separate receiver-proof repair are
accepted.
