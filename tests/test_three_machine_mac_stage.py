from pathlib import Path

from experiments.three_machine import mac_stage, protocol


class FakeStage:
    context_capacity = 2051

    def __init__(self):
        self.forwards = []
        self.commits = []
        self.resets = 0

    def forward(self, tokens):
        self.forwards.append(list(tokens))
        return b"\0" * (len(tokens) * protocol.ROW_BYTES)

    def commit(self, keep=None):
        self.commits.append(keep)

    def reset(self):
        self.resets += 1


class FakeTokenizer:
    eos_token_ids = []

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ",".join(str(token) for token in tokens)


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
        if frame.kind is protocol.Kind.WINDOW_ACTIVATION:
            tokens, activations = protocol.window_activation_values(frame)
            assert len(activations) == len(tokens) * protocol.ROW_BYTES
            self.windows += 1
            if self.windows == 1:
                assert tokens == [10, 11]
                accepted, drafts, keep = [100], [200, 201], 2
            else:
                assert tokens == [100, 200]
                accepted, drafts, keep = [101, 102], [202], 2
            return protocol.pack(protocol.window_result(
                request_id=frame.request_id,
                step=frame.step,
                committed=frame.committed + keep,
                generation=frame.generation,
                accepted=accepted,
                drafts=drafts,
                timeout_s=1,
            ))
        if frame.kind is protocol.Kind.ACK:
            return protocol.pack(frame)
        if frame.kind is protocol.Kind.RESET:
            return protocol.pack(protocol.control(
                kind=protocol.Kind.ACK,
                request_id=frame.request_id,
                step=0,
                committed=0,
                generation=frame.generation,
                timeout_s=1,
            ))
        raise AssertionError(f"unexpected frame {frame.kind}")


def test_mac_stage_runs_prompt_decode_commit_and_final_reset(monkeypatch):
    monkeypatch.setattr(mac_stage, "ClientMailbox", FakeMailbox)
    stage = FakeStage()

    result = mac_stage.run(
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
    assert result["generation"] == 7
    assert result["committed"] == 4
    assert len(result["steps"]) == 2
    assert stage.forwards == [[10, 11], [100, 200]]
    assert stage.commits == [2, 2]
    assert stage.resets == 2
    assert result["decode_rounds"] == 1
    assert result["tokens_per_decode_round"] == 2
    assert result["draft_acceptance_rate"] == 1
