import ctypes
import mmap
import os
import socket
import tempfile
import threading
import time

import pytest

from experiments.three_machine.mailbox import (
    CTRL,
    GENERATION_WORD,
    READY_WORD,
    SIZES,
    STAGED_WORD,
    ClientMailbox,
    MailboxError,
    ServiceMailbox,
    _Mapped,
    link_generation,
)


class FakeHelper:
    @staticmethod
    def mcdma_rpc_store_word(address, value):
        ctypes.c_uint64.from_address(address).value = value

    @staticmethod
    def mcdma_rpc_wait_word(address, seq, equal, _spin_ns, timeout_ns):
        deadline = time.monotonic_ns() + timeout_ns
        while time.monotonic_ns() < deadline:
            value = ctypes.c_uint64.from_address(address).value
            current = value >> 32
            if (equal and current == seq) or (not equal and current and current != seq):
                return value
            time.sleep(0.0001)
        return 0


def mailbox_file(*, ready=1, generation=7):
    handle = tempfile.NamedTemporaryFile(delete=False)
    handle.truncate(8 << 20)
    handle.close()
    with open(handle.name, "r+b") as stream:
        box = mmap.mmap(stream.fileno(), 0)
        box[SIZES:SIZES + 8] = (4 << 20).to_bytes(8, "little")
        box[SIZES + 8:SIZES + 16] = (4 << 20).to_bytes(8, "little")
        box[READY_WORD:READY_WORD + 8] = int(ready).to_bytes(8, "little")
        box[GENERATION_WORD:GENERATION_WORD + 8] = int(generation).to_bytes(8, "little")
        box.close()
    return handle.name


def test_client_call_uses_exact_mailbox_halves():
    path = mailbox_file()
    try:
        def peer():
            with open(path, "r+b") as stream:
                box = mmap.mmap(stream.fileno(), 0)
                while int.from_bytes(box[0:8], "little") == 0:
                    time.sleep(0.0001)
                word = int.from_bytes(box[0:8], "little")
                seq, length = word >> 32, word & 0xFFFFFFFF
                payload = bytes(box[CTRL:CTRL + length])
                response = payload[::-1]
                start = (4 << 20) + CTRL
                box[start:start + len(response)] = response
                offset = (4 << 20) + READY_WORD
                box[offset:offset + 8] = ((seq << 32) | len(response)).to_bytes(8, "little")
                box.close()

        thread = threading.Thread(target=peer, daemon=True)
        thread.start()
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            assert client.generation == 7
            assert client.call(b"three-machine", 1.0) == b"enihcam-eerht"
        thread.join(1)
        assert not thread.is_alive()
    finally:
        os.unlink(path)


def test_client_can_consume_a_reply_in_place():
    path = mailbox_file()
    try:
        def peer():
            with open(path, "r+b") as stream:
                box = mmap.mmap(stream.fileno(), 0)
                while int.from_bytes(box[0:8], "little") == 0:
                    time.sleep(0.0001)
                word = int.from_bytes(box[0:8], "little")
                seq = word >> 32
                response = b"mapped-reply"
                start = (4 << 20) + CTRL
                box[start:start + len(response)] = response
                offset = (4 << 20) + READY_WORD
                box[offset:offset + 8] = ((seq << 32) | len(response)).to_bytes(8, "little")
                box.close()

        observed = {}

        def consume(reply):
            observed["is_memoryview"] = isinstance(reply, memoryview)
            observed["readonly"] = reply.readonly
            observed["value"] = bytes(reply)
            return len(reply)

        thread = threading.Thread(target=peer, daemon=True)
        thread.start()
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            assert client.call_consume(b"request", consume, 1.0) == len(b"mapped-reply")
        thread.join(1)
        assert not thread.is_alive()
        assert observed == {
            "is_memoryview": True,
            "readonly": True,
            "value": b"mapped-reply",
        }
    finally:
        os.unlink(path)


def test_client_refuses_generation_change():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            with open(path, "r+b") as stream:
                box = mmap.mmap(stream.fileno(), 0)
                box[GENERATION_WORD:GENERATION_WORD + 8] = (8).to_bytes(8, "little")
                box.close()
            with pytest.raises(MailboxError, match="generation changed"):
                client.call(b"request", 0.01)
    finally:
        os.unlink(path)


def test_link_generation_probe_is_read_only_and_requires_ready_link():
    path = mailbox_file(generation=11)
    try:
        assert link_generation("test", mailbox_path=path) == 11
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[READY_WORD:READY_WORD + 8] = (0).to_bytes(8, "little")
            box.close()
        with pytest.raises(MailboxError, match="not ready"):
            link_generation("test", mailbox_path=path)
    finally:
        os.unlink(path)


def test_client_burns_timed_out_sequence_before_retry():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            first_seq = client._seq
            with pytest.raises(MailboxError, match="timed out"):
                client.call(b"first", 0.01)
            assert client._seq != first_seq

            def peer():
                with open(path, "r+b") as stream:
                    box = mmap.mmap(stream.fileno(), 0)
                    while True:
                        word = int.from_bytes(box[0:8], "little")
                        seq, length = word >> 32, word & 0xFFFFFFFF
                        if seq != first_seq:
                            break
                        time.sleep(0.0001)
                    payload = bytes(box[CTRL:CTRL + length])
                    response = b"reply:" + payload
                    start = (4 << 20) + CTRL
                    box[start:start + len(response)] = response
                    offset = (4 << 20) + READY_WORD
                    box[offset:offset + 8] = ((seq << 32) | len(response)).to_bytes(8, "little")
                    box.close()

            thread = threading.Thread(target=peer, daemon=True)
            thread.start()
            assert client.call(b"second", 1.0) == b"reply:second"
            thread.join(1)
            assert not thread.is_alive()
    finally:
        os.unlink(path)


def test_client_seeds_away_from_request_and_done_words():
    path = mailbox_file()
    try:
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[0:8] = ((3 << 32) | 1).to_bytes(8, "little")
            done = (4 << 20) + READY_WORD
            box[done:done + 8] = ((4 << 32) | 1).to_bytes(8, "little")
            box.close()
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            assert client._seq not in {0, 3, 4}
    finally:
        os.unlink(path)


def test_client_rejects_a_helper_result_for_another_sequence():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            pending = client._seq
            client._wait = lambda *_args: (((pending + 1) << 32) | 1)
            with pytest.raises(MailboxError, match="sequence differs"):
                client.call(b"request", 0.1)
    finally:
        os.unlink(path)


def test_client_rejects_reply_control_word_changed_during_copy():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            pending = client._seq
            done = (pending << 32) | 5
            start = client.request_bytes + CTRL
            client.buffer[start:start + 5] = b"reply"
            client._wait = lambda *_args: done
            original_load = client._load

            def changed(offset):
                if offset == client.request_bytes + READY_WORD:
                    return ((pending + 1) << 32) | 5
                return original_load(offset)

            client._load = changed
            with pytest.raises(MailboxError, match="changed while it was copied"):
                client.call(b"request", 0.1)
    finally:
        os.unlink(path)


def test_client_sequence_seed_changes_with_link_generation():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as first:
            first_sequence = first._seq
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[GENERATION_WORD:GENERATION_WORD + 8] = (8).to_bytes(8, "little")
            box.close()
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as second:
            assert second._seq != first_sequence
    finally:
        os.unlink(path)


def test_client_rejects_nonpositive_timeout_before_publish():
    path = mailbox_file()
    try:
        with ClientMailbox("test", mailbox_path=path, helper=FakeHelper()) as client:
            with pytest.raises(MailboxError, match="timeout must be positive"):
                client.call(b"request", 0)
            with open(path, "rb") as stream:
                assert int.from_bytes(stream.read(8), "little") == 0
    finally:
        os.unlink(path)


def service_mailbox(path):
    service = ServiceMailbox.__new__(ServiceMailbox)
    _Mapped.__init__(service, "test", mailbox_path=path, helper=FakeHelper())
    service._control, daemon = socket.socketpair()
    service.generation = None
    service._last = service._load(0) >> 32
    return service, daemon


def test_service_binds_authenticated_frame_generation_not_local_control_words():
    path = mailbox_file()
    service = daemon = None
    try:
        service, daemon = service_mailbox(path)
        assert service.generation is None
        service.bind_generation(7)
        request = b"frame"
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[CTRL:CTRL + len(request)] = request
            box[0:8] = ((1 << 32) | len(request)).to_bytes(8, "little")
            box.close()
        assert service.next_request(0.2) == (1, request)
        service.reply(1, b"answer")
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            start = (4 << 20) + CTRL
            assert bytes(box[start:start + 6]) == b"answer"
            staged = (4 << 20) + STAGED_WORD
            assert int.from_bytes(box[staged:staged + 8], "little") == (1 << 32) | 6
            accepted = 4 << 20
            box[accepted:accepted + 8] = ((1 << 32) | 6).to_bytes(8, "little")
            box.close()
        service.wait_reply_accepted(1, 6, 0.1)
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[READY_WORD:READY_WORD + 8] = (0).to_bytes(8, "little")
            box[GENERATION_WORD:GENERATION_WORD + 8] = (0).to_bytes(8, "little")
            box.close()
        assert service.next_request(0.01) is None
        service.reply(2, b"answer")
        service.bind_generation(7)
        with pytest.raises(MailboxError, match="different MCDMA generation"):
            service.bind_generation(8)
    finally:
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_service_can_publish_a_reply_written_in_place():
    path = mailbox_file()
    service = daemon = area = None
    try:
        service, daemon = service_mailbox(path)
        area = service.reply_area()
        assert area.nbytes == service.max_reply
        area[:6] = b"answer"
        service.publish(9, 6)
        area.release()
        area = None
        with open(path, "rb") as stream:
            box = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                start = (4 << 20) + CTRL
                assert bytes(box[start:start + 6]) == b"answer"
                staged = (4 << 20) + STAGED_WORD
                assert int.from_bytes(box[staged:staged + 8], "little") == (9 << 32) | 6
            finally:
                box.close()
        with pytest.raises(MailboxError, match="exceeds"):
            service.publish(10, service.max_reply + 1)
    finally:
        if area is not None:
            area.release()
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_real_service_registration_accepts_zero_linux_ready_and_generation_words():
    path = mailbox_file(ready=0, generation=0)
    stopped = threading.Event()
    try:
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "rpcd.sock")
            listening = threading.Event()

            def daemon():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                    server.bind(socket_path)
                    server.listen(1)
                    listening.set()
                    connection, _ = server.accept()
                    with connection:
                        assert connection.recv(64) == b"MODE poll\n"
                        connection.sendall(b"OK\n")
                        stopped.wait(2)

            thread = threading.Thread(target=daemon, daemon=True)
            thread.start()
            assert listening.wait(1)
            with ServiceMailbox(
                    "test", socket_path=socket_path, mailbox_path=path,
                    helper=FakeHelper()) as service:
                assert service.generation is None
                service.bind_generation(11)
                assert service.generation == 11
            stopped.set()
            thread.join(1)
            assert not thread.is_alive()
    finally:
        stopped.set()
        os.unlink(path)


def test_service_detects_control_disconnect_and_empty_messages():
    path = mailbox_file()
    service = daemon = None
    try:
        service, daemon = service_mailbox(path)
        with pytest.raises(MailboxError, match="reply is empty"):
            service.reply(1, b"")
        daemon.close()
        daemon = None
        with pytest.raises(MailboxError, match="control connection closed"):
            service.next_request(0.1)
    finally:
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_service_consumes_a_malformed_word_before_rejecting_it():
    path = mailbox_file()
    service = daemon = None
    try:
        service, daemon = service_mailbox(path)
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[0:8] = ((1 << 32) | 0xFFFFFFFF).to_bytes(8, "little")
            box.close()
        with pytest.raises(MailboxError, match="exceeds"):
            service.next_request(0.1)
        assert service._last == 1

        request = b"valid"
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[CTRL:CTRL + len(request)] = request
            box[0:8] = ((2 << 32) | len(request)).to_bytes(8, "little")
            box.close()
        assert service.next_request(0.1) == (2, request)
    finally:
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_service_ignores_a_request_left_before_registration():
    path = mailbox_file()
    service = daemon = None
    try:
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[0:8] = ((9 << 32) | 5).to_bytes(8, "little")
            box.close()
        service, daemon = service_mailbox(path)
        assert service._last == 9
        assert service.next_request(0.01) is None
    finally:
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_service_discards_request_rewritten_during_copy():
    path = mailbox_file()
    service = daemon = None
    try:
        service, daemon = service_mailbox(path)
        first = b"first"
        second = b"second-stable"
        with open(path, "r+b") as stream:
            box = mmap.mmap(stream.fileno(), 0)
            box[CTRL:CTRL + len(first)] = first
            box[0:8] = ((1 << 32) | len(first)).to_bytes(8, "little")
            box.close()
        original_load = service._load
        rewritten = False

        def rewrite_on_validation(offset):
            nonlocal rewritten
            if offset == 0 and not rewritten:
                rewritten = True
                with open(path, "r+b") as stream:
                    box = mmap.mmap(stream.fileno(), 0)
                    box[CTRL:CTRL + len(second)] = second
                    box[0:8] = ((2 << 32) | len(second)).to_bytes(8, "little")
                    box.close()
                return (2 << 32) | len(second)
            return original_load(offset)

        service._load = rewrite_on_validation
        assert service.next_request(0.2) == (2, second)
    finally:
        if service is not None:
            service.close()
        if daemon is not None:
            daemon.close()
        os.unlink(path)


def test_service_constructor_performs_poll_registration_handshake():
    path = mailbox_file(ready=0, generation=0)
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        socket_path = os.path.join(temporary, "mcdma.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        received = []
        try:
            server.bind(socket_path)
            server.listen(1)

            def peer():
                connection, _ = server.accept()
                with connection:
                    received.append(connection.recv(64))
                    connection.sendall(b"OK\n")

            thread = threading.Thread(target=peer, daemon=True)
            thread.start()
            with ServiceMailbox(
                "test", socket_path=socket_path, mailbox_path=path, helper=FakeHelper()
            ) as service:
                assert service.generation is None
                service.bind_generation(7)
                assert service.generation == 7
            thread.join(1)
            assert received == [b"MODE poll\n"]
        finally:
            server.close()
            os.unlink(path)
