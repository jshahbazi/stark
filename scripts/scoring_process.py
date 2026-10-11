"""Standard-library ownership adapter for the private scoring validator.

The native wrapper retains the unreaped child identity until group cleanup.
Python requests stop through its control socket and never signals a stored PID.
Wrapper loss remains an unsettled instance, never a successful cleanup claim.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
import socket
import struct
import uuid


class ProcessError(RuntimeError):
    pass


async def settle(action):
    task = asyncio.ensure_future(action)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class WriteProtocol(asyncio.streams.FlowControlMixin):
    def __init__(self, loop):
        super().__init__(loop=loop)
        self.closed = loop.create_future()

    def connection_lost(self, exc):
        super().connection_lost(exc)
        if not self.closed.done():
            self.closed.set_result(None)

    def _get_close_waiter(self, _stream):
        return self.closed


class OwnedProcess:
    def __init__(self, owner, identifier, wrapper, control, input_fd, output_fd, grace):
        self.owner, self.identifier, self.wrapper = owner, identifier, wrapper
        self.control, self.input_fd, self.output_fd, self.grace = control, input_fd, output_fd, grace
        self.stdin, self.stdout, self.transport = None, None, None
        self.returncode, self.ownership_lost = None, False
        loop = asyncio.get_running_loop()
        self.spawned, self.done = loop.create_future(), loop.create_future()
        self.monitor = asyncio.create_task(self.observe())

    async def observe(self):
        terminal = False
        spawned = False
        peak = False
        buffer = bytearray()
        try:
            while chunk := await asyncio.get_running_loop().sock_recv(self.control, 4096):
                buffer.extend(chunk)
                if len(buffer) > 8192:
                    raise ProcessError("ownership_protocol_invalid")
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    event = json.loads(line)
                    value = event["value"]
                    if type(value) is not int or set(event) != {"event", "value"} or terminal:
                        raise ProcessError("ownership_protocol_invalid")
                    if event["event"] == "spawn" and not spawned and value > 1:
                        spawned = True
                        self.spawned.set_result(True)
                    elif event["event"] == "peak_memory" and not peak and 0 <= value <= 2**63 - 1:
                        peak = True
                    elif event["event"] in {"exit", "launch_failed"}:
                        terminal = True
                        self.returncode = value
                        if not self.spawned.done():
                            self.spawned.set_result(False)
                    else:
                        raise ProcessError("ownership_protocol_invalid")
            if buffer:
                terminal = False
        except Exception:
            terminal = False
        finally:
            self.control.close()
            wrapper_code = await self.wrapper.wait()
            self.ownership_lost = not terminal or wrapper_code != 0
            if not self.spawned.done():
                self.spawned.set_result(False)
            if not self.ownership_lost:
                self.owner.instances.pop(self.identifier, None)
            self.done.set_result(None)

    async def streams(self):
        loop = asyncio.get_running_loop()
        self.stdout = asyncio.StreamReader(limit=1_048_576)
        read_protocol = asyncio.StreamReaderProtocol(self.stdout)
        read_pipe = os.fdopen(self.output_fd, "rb", buffering=0)
        self.output_fd = None
        self.transport, _ = await loop.connect_read_pipe(lambda: read_protocol, read_pipe)
        protocol = WriteProtocol(loop)
        write_pipe = os.fdopen(self.input_fd, "wb", buffering=0)
        self.input_fd = None
        transport, _ = await loop.connect_write_pipe(lambda: protocol, write_pipe)
        self.stdin = asyncio.StreamWriter(transport, protocol, None, loop)

    async def wait(self):
        await asyncio.shield(self.done)
        if self.ownership_lost:
            raise ProcessError("ownership_lost")
        return self.returncode

    async def stop(self):
        if self.stdin is not None:
            self.stdin.close()
        if self.transport is not None:
            self.transport.close()
        for attribute in ("input_fd", "output_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(self, attribute, None)
        if not self.done.done():
            with contextlib.suppress(OSError):
                await asyncio.get_running_loop().sock_sendall(self.control, b"S")
        async with asyncio.timeout(self.grace + 5):
            return await self.wait()


class ProcessOwner:
    def __init__(self, reaper):
        self.reaper = Path(reaper).absolute()
        self.instances, self.launches, self.closed = {}, set(), False

    async def launch(self, executable, arguments, cwd, environment, *, io_mode="streams", owned_group=True, stop_grace=.5):
        if (self.closed or self.instances or self.launches or io_mode != "streams"
                or owned_group is not True or not 0 <= stop_grace <= 30):
            raise ProcessError("invalid_launch")
        task = asyncio.create_task(self._launch(executable, arguments, cwd, environment, stop_grace))
        self.launches.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            process = await settle(task)
            await settle(process.stop())
            raise
        finally:
            self.launches.discard(task)

    async def _launch(self, executable, arguments, cwd, environment, grace):
        payload = bytearray(b"STARK003")
        spec_parent, spec_child = socket.socketpair()
        control_parent, control_child = socket.socketpair()
        input_read, input_write = os.pipe()
        output_read, output_write = os.pipe()
        process = None
        def number(value):
            payload.extend(struct.pack("!I", value))
        def string(value):
            if not isinstance(value, str) or "\0" in value:
                raise ProcessError("invalid_launch")
            data = value.encode()
            number(len(data))
            payload.extend(data)
        try:
            for value in (int(grace * 1000), input_read, output_write, 0):
                number(value)
            string(str(executable)); string(str(cwd)); number(len(arguments))
            for value in arguments:
                string(value)
            number(len(environment))
            for key, value in environment.items():
                if not key or "=" in key:
                    raise ProcessError("invalid_launch")
                string(key + "=" + value)
            if len(payload) > 1_048_576:
                raise ProcessError("invalid_launch")
            wrapper = await asyncio.create_subprocess_exec(
                str(self.reaper), str(spec_child.fileno()), str(control_child.fileno()), str(os.getpid()),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env={}, pass_fds=(spec_child.fileno(), control_child.fileno(), input_read, output_write))
            control_parent.setblocking(False)
            identifier = uuid.uuid4().hex
            process = OwnedProcess(self, identifier, wrapper, control_parent, input_write, output_read, grace)
            self.instances[identifier] = process
            spec_child.close(); control_child.close()
            os.close(input_read); input_read = -1
            os.close(output_write); output_write = -1
            spec_parent.setblocking(False)
            async with asyncio.timeout(10):
                await asyncio.get_running_loop().sock_sendall(spec_parent, payload)
                spec_parent.close()
                if not await asyncio.shield(process.spawned):
                    await process.wait()
                    raise ProcessError("launch_failed")
            await process.streams()
            if self.closed:
                raise ProcessError("owner_closed")
            return process
        except BaseException:
            if process is not None:
                await settle(process.stop())
            raise
        finally:
            spec_parent.close(); spec_child.close(); control_child.close()
            for descriptor in (input_read, output_write):
                if descriptor >= 0:
                    os.close(descriptor)
            if process is None:
                control_parent.close()
                os.close(input_write); os.close(output_read)

    async def close(self):
        self.closed = True
        if self.launches:
            await settle(asyncio.gather(*self.launches, return_exceptions=True))
        results = await settle(asyncio.gather(*(process.stop() for process in list(self.instances.values())), return_exceptions=True))
        if self.instances or any(isinstance(value, BaseException) for value in results):
            raise ProcessError("ownership_unsettled")
