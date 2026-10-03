from pathlib import Path

from experiments.three_machine import mac_drafter, protocol


class FakeStage:
    context_capacity = 2051
    draft_count = 3

    def __init__(self):
        self.prompt_calls = []
        self.decode_calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def absorb_prompt(self, tokens, hidden, accepted, count):
        self.prompt_calls.append((list(tokens), len(hidden), list(accepted), count))
        return [200, 201]

    def absorb_decode(self, hidden, accepted, count):
        self.decode_calls.append((len(hidden), list(accepted), count))
        return []


class FakeTokenizer:
    eos_token_ids = []

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ",".join(map(str, tokens))


class FakeMailbox:
    generation = 7

    def __init__(self, _name, ready_timeout_s):
        assert ready_timeout_s == 1
        self.windows = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def call(self, raw, _timeout):
        frame = protocol.unpack(raw)
        if frame.kind is protocol.Kind.RESET:
            return protocol.pack(protocol.control(
                kind=protocol.Kind.ACK,
                request_id=frame.request_id,
                step=0,
                committed=0,
                generation=frame.generation,
                timeout_s=1,
            ))
        if frame.kind is protocol.Kind.ACK:
            return protocol.pack(frame)
        assert frame.kind is protocol.Kind.TOKEN_WINDOW
        tokens = protocol.token_window_values(frame)
        self.windows += 1
        if self.windows == 1:
            assert tokens == [10, 11]
            assert frame.flags == protocol.Flags.PROMPT | protocol.Flags.FINAL
            accepted, rows, keep = [100], 2, 2
        else:
            assert tokens == [100, 200]
            assert frame.flags == protocol.Flags.DECODE
            accepted, rows, keep = [101, 102], 2, 2
        return protocol.pack(protocol.verified_hidden(
            request_id=frame.request_id,
            step=frame.step,
            committed=frame.committed + keep,
            generation=frame.generation,
            accepted=accepted,
            hidden=b"\0" * (rows * protocol.ROW_BYTES),
            rows=rows,
            prompt=bool(frame.flags & protocol.Flags.PROMPT),
            final=bool(frame.flags & protocol.Flags.FINAL),
            timeout_s=1,
        ))


def test_mac_drafter_runs_verified_prompt_decode_and_reset(monkeypatch):
    monkeypatch.setattr(mac_drafter, "ClientMailbox", FakeMailbox)
    stage = FakeStage()

    result = mac_drafter.run(
        Path("/unused"),
        "unused",
        3,
        "test",
        1,
        False,
        stage=stage,
        tokenizer=FakeTokenizer(),
        prompt_tokens=[10, 11],
    )

    assert result["generated_tokens"] == [100, 101, 102]
    assert result["text"] == "100,101,102"
    assert result["committed"] == 4
    assert result["decode_rounds"] == 1
    assert result["verified_drafts"] == 1
    assert result["accepted_drafts"] == 1
    assert result["draft_acceptance_rate"] == 1
    assert stage.prompt_calls == [([10, 11], 2 * protocol.ROW_BYTES, [100], 2)]
    assert stage.decode_calls == [(2 * protocol.ROW_BYTES, [101, 102], 0)]
    assert stage.resets == 2
