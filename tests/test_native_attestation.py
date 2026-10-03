import copy

import pytest

from experiments.three_machine.native_attestation import (
    NativeAttestationError,
    attest,
    attest_replay,
)


IMAGE = "sha256:" + "a" * 64
MODEL = "/models/glm"
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"


def container(role):
    return {
        "Id": role + "-id",
        "Image": IMAGE,
        "Path": "python3",
        "Args": ["-m", "tensorfold"],
        "Config": {
            "Cmd": [
                "python3", "-m", "tensorfold", "serve", MODEL,
                "--backend", "cuda", "--tp", "2", "--parallel", "4",
                "--drafter", DRAFTER, "--rank", "0" if role == "head" else "1",
            ],
            "Env": ["TF_GLM_DENSE=q4"],
            "Labels": {"glm.mcdma.mode": "native-portable"},
        },
        "HostConfig": {"NetworkMode": "host", "IpcMode": "host"},
        "Mounts": [{"Source": "/models", "Destination": "/models", "RW": False}],
        "RestartCount": 0,
        "State": {
            "Running": True,
            "OOMKilled": False,
            "StartedAt": "2026-10-02T00:00:00Z",
        },
    }


def evidence(**changes):
    values = {
        "anchor_head": container("head"),
        "anchor_worker": container("worker"),
        "before_head": container("head"),
        "after_head": container("head"),
        "before_worker": container("worker"),
        "after_worker": container("worker"),
        "health_before": {"status": "ready"},
        "health_after": {"status": "ready"},
        "expected_image": IMAGE,
        "expected_parallel": 4,
        "expected_model": MODEL,
        "expected_drafter": DRAFTER,
    }
    values.update(changes)
    return values


def test_accepts_exact_unchanged_container_configuration():
    result = attest(**evidence())
    assert result["accepted"] is True
    assert result["head"]["restart_count"] == 0
    assert len(result["head"]["configuration_sha256"]) == 64


def test_rejects_restart_and_any_configuration_change():
    restarted = container("head")
    restarted["RestartCount"] = 1
    with pytest.raises(NativeAttestationError, match="restart count"):
        attest(**evidence(after_head=restarted))

    changed = copy.deepcopy(container("worker"))
    changed["Config"]["Env"].append("UNREVIEWED=1")
    with pytest.raises(NativeAttestationError, match="configuration changed"):
        attest(**evidence(after_worker=changed))


def test_accepts_docker_mount_projection_in_a_different_order():
    anchor = container("head")
    anchor["Mounts"].append(
        {"Source": "/cache", "Destination": "/cache", "RW": True}
    )
    observed = copy.deepcopy(anchor)
    observed["Mounts"].reverse()

    result = attest(
        **evidence(
            anchor_head=anchor,
            before_head=observed,
            after_head=copy.deepcopy(observed),
        )
    )

    assert result["head"]["container_id"] == "head-id"


def test_accepts_unique_docker_environment_projection_in_a_different_order():
    anchor = container("head")
    anchor["Config"]["Env"].append("TF_EXTRA=1")
    observed = copy.deepcopy(anchor)
    observed["Config"]["Env"].reverse()

    result = attest(
        **evidence(
            anchor_head=anchor,
            before_head=observed,
            after_head=copy.deepcopy(observed),
        )
    )

    assert result["head"]["configuration_sha256"]


def test_preserves_order_for_duplicate_docker_environment_keys():
    anchor = container("head")
    anchor["Config"]["Env"] = ["TF_DUPLICATE=first", "TF_DUPLICATE=last"]
    observed = copy.deepcopy(anchor)
    observed["Config"]["Env"].reverse()

    with pytest.raises(NativeAttestationError, match="configuration changed"):
        attest(
            **evidence(
                anchor_head=anchor,
                before_head=observed,
                after_head=copy.deepcopy(observed),
            )
        )


def test_rejects_unready_health_and_wrong_command():
    with pytest.raises(NativeAttestationError, match="health"):
        attest(**evidence(health_after={"status": "loading"}))

    wrong = copy.deepcopy(container("head"))
    command = wrong["Config"]["Cmd"]
    command[command.index("--parallel") + 1] = "8"
    with pytest.raises(NativeAttestationError, match="lane count"):
        attest(**evidence(before_head=wrong))

    wrong_rank = copy.deepcopy(container("worker"))
    command = wrong_rank["Config"]["Cmd"]
    command[command.index("--rank") + 1] = "0"
    with pytest.raises(NativeAttestationError, match="distributed rank"):
        attest(**evidence(before_worker=wrong_rank))


def replay_container(prompt="c" * 64):
    value = container("replay")
    value["Config"]["Cmd"] = [
        "python3", "-m", "experiments.three_machine.cache_replay", "--bundle", prompt,
    ]
    value["Config"]["Labels"] = {
        "glm.mcdma.mode": "cache-replay",
        "glm.mcdma.prompt": prompt,
    }
    return value


def test_replay_attestation_binds_identity_image_start_and_prompt():
    prompt = "c" * 64
    result = attest_replay(
        anchor=replay_container(prompt),
        before=replay_container(prompt),
        after=replay_container(prompt),
        expected_image=IMAGE,
        expected_prompt=prompt,
    )
    assert result["container_id"] == "replay-id"
    assert result["prompt_sha256"] == prompt

    changed = replay_container("d" * 64)
    with pytest.raises(NativeAttestationError, match="prompt label"):
        attest_replay(
            anchor=replay_container(prompt),
            before=changed,
            after=replay_container(prompt),
            expected_image=IMAGE,
            expected_prompt=prompt,
        )


def test_stable_configuration_hash_ignores_only_run_identity():
    first = container("head")
    first["Config"]["Hostname"] = "container-one"
    first["Config"]["Labels"]["glm.mcdma.run"] = "run-one"
    second = copy.deepcopy(first)
    second["Config"]["Hostname"] = "container-two"
    second["Config"]["Labels"]["glm.mcdma.run"] = "run-two"
    second["Id"] = first["Id"]
    second["State"]["StartedAt"] = first["State"]["StartedAt"]
    result = attest(**evidence(
        anchor_head=first, before_head=first, after_head=first,
    ))
    other = attest(**evidence(
        anchor_head=second, before_head=second, after_head=second,
    ))
    assert result["head"]["configuration_sha256"] == other["head"]["configuration_sha256"]
    assert result["head"]["runtime_configuration_sha256"] != other["head"]["runtime_configuration_sha256"]
