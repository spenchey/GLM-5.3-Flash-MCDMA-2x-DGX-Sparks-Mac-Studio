import pytest

from experiments.three_machine import cuda_main, protocol


class Done(Exception):
    pass


class SequenceMailbox:
    def __init__(self, frames):
        self.frames = list(frames)
        self.generation = None
        self.replies = []
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def next_request(self, _timeout):
        if not self.frames:
            raise Done
        self.calls += 1
        return self.calls, protocol.pack(self.frames.pop(0))

    def bind_generation(self, generation):
        if self.generation is None:
            self.generation = generation
        assert self.generation == generation

    def reply(self, sequence, raw):
        self.replies.append((sequence, protocol.unpack(raw)))


def test_rank_zero_runs_complete_main_window_ack_reset_and_shutdown(monkeypatch):
    generation = 7
    request_id = 19
    frames = [
        protocol.token_window(
            request_id=request_id,
            step=0,
            committed=0,
            generation=generation,
            tokens=[10, 11],
            prompt=True,
            final=True,
            timeout_s=2,
        ),
        protocol.control(
            kind=protocol.Kind.ACK,
            request_id=request_id,
            step=0,
            committed=2,
            generation=generation,
            timeout_s=2,
        ),
        protocol.control(
            kind=protocol.Kind.RESET,
            request_id=request_id,
            step=1,
            committed=2,
            generation=generation,
            timeout_s=2,
        ),
        protocol.control(
            kind=protocol.Kind.SHUTDOWN,
            request_id=request_id,
            step=0,
            committed=0,
            generation=generation,
            timeout_s=2,
        ),
    ]
    mailbox = SequenceMailbox(frames)
    controls = []
    monkeypatch.setattr(cuda_main, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_main, "_control", lambda _comm, _rank, values: controls.append(values))
    monkeypatch.setattr(cuda_main, "_token_ids", lambda _comm, _rank, tokens, _rows: list(tokens))
    monkeypatch.setattr(cuda_main, "_rank_memory", lambda: {
        "cuda_allocated_bytes": 1,
        "cuda_reserved_bytes": 2,
        "process_rss_bytes": 3,
    })

    class Engine:
        comm = object()
        context_capacity = 2051
        last_run_metrics = {"gpu_total_s": 0.01, "verified_drafts": 0, "accepted_drafts": 0}

        def __init__(self):
            self.acks = []
            self.resets = 0
            self.rings = 0

        def ring(self):
            self.rings += 1

        def process(self, tokens, *, prompt, final):
            assert tokens == [10, 11]
            assert prompt is True and final is True
            return cuda_main.MainRun([42], b"h" * (2 * protocol.ROW_BYTES), 2)

        def ack(self, rows, keep):
            self.acks.append((rows, keep))

        def reset(self):
            self.resets += 1

    engine = Engine()
    result = cuda_main.rank_zero(engine, "test", "/tmp/test.sock", 2)

    assert result == {"rank": 0, "committed_windows": 1, "committed_tokens": 0}
    assert engine.acks == [(2, 2)]
    assert engine.resets == 1
    assert engine.rings == 4
    assert controls == [
        (cuda_main.ACTION_TOKEN_WINDOW, 2, int(protocol.Flags.PROMPT | protocol.Flags.FINAL), 0, 0, request_id),
        (cuda_main.ACTION_ACK, 2, 2, 0, 2, request_id),
        (cuda_main.ACTION_RESET, 0, 0, 0, 0, request_id),
        (cuda_main.ACTION_SHUTDOWN, 0, 0, 0, 0, request_id),
    ]
    verified = mailbox.replies[0][1]
    assert verified.kind is protocol.Kind.VERIFIED_HIDDEN
    assert protocol.verified_hidden_values(verified)[0] == [42]


def test_rank_zero_rejects_capacity_overflow_before_collective(monkeypatch):
    frame = protocol.token_window(
        request_id=9,
        step=0,
        committed=2050,
        generation=7,
        tokens=[1, 2],
        prompt=False,
        timeout_s=2,
    )
    mailbox = SequenceMailbox([frame])
    controls = []
    monkeypatch.setattr(cuda_main, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_main, "_control", lambda _comm, _rank, values: controls.append(values))

    class ExistingSession:
        def __init__(self, generation):
            self.generation = generation
            self.request_id = 9
            self.step = 0
            self.committed = 2050

        def classify_activation(self, _frame):
            return None

    monkeypatch.setattr(cuda_main, "Session", ExistingSession)
    engine = type("Engine", (), {
        "comm": object(),
        "context_capacity": 2051,
        "ring": lambda self: None,
    })()

    with pytest.raises(Done):
        cuda_main.rank_zero(engine, "test", "/tmp/test.sock", 2)

    assert controls == [(cuda_main.ACTION_NOOP, 0, 0, 0, 0, 0)]
    assert mailbox.replies[0][1].kind is protocol.Kind.ERROR
    assert b"supports 2051" in mailbox.replies[0][1].payload


def test_rank_one_waits_for_store_doorbell_before_runtime_collective(monkeypatch):
    events = []

    class Engine:
        comm = object()
        e = type("Runtime", (), {"st": type("State", (), {"pos": 0})()})()

        def await_action(self):
            events.append("doorbell")

    monkeypatch.setattr(
        cuda_main,
        "_control",
        lambda _comm, rank: events.append(("control", rank))
        or (cuda_main.ACTION_SHUTDOWN, 0, 0, 0, 0, 29),
    )

    result = cuda_main.rank_one(Engine())

    assert events == ["doorbell", ("control", 1)]
    assert result == {"rank": 1, "committed_windows": 0, "committed_tokens": 0}
