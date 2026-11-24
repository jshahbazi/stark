"""Decode the publisher's framed assistant output without exposing tool payloads."""

import json
import re

PARSER = "inkling2-json-v1"
_MODEL = "<|message_model|>"
_THINKING = "<|content_thinking|>"
_TEXT = "<|content_text|>"
_TOOL = "<|content_invoke_tool_json|>"
_TOOL_TEXT = "<|content_invoke_tool_text|>"
_TOOL_ERROR = "<|content_tool_error|>"
_END = "<|end_message|>"
_SAMPLE_END = "<|content_model_end_sampling|>"
_BOUNDARIES = ("", _END, _SAMPLE_END)
_FRAMES = (_MODEL, _THINKING, _TEXT, _TOOL, _TOOL_TEXT, _TOOL_ERROR, _END, _SAMPLE_END)
_MARKER = re.compile("|".join(map(re.escape, _FRAMES)))
_MAX_MARKER = max(map(len, _FRAMES))


class OutputFrames:
    def __init__(self, content, thinking):
        self.content = content
        self.thinking = thinking
        self.mode = ""
        self.tail = ""
        self.name = ""
        self.awaiting_final = False
        self.invalid_call = False
        self.calls = []

    def feed(self, text):
        if not self.awaiting_final:
            self._consume(text, final=False)

    def parse(self, text):
        self._consume(text, final=True)

    def _consume(self, text, *, final):
        value, self.tail = self.tail + text, ""
        start = 0
        while start < len(value):
            if self.mode == _TOOL:
                if not final:
                    self.awaiting_final = True
                    return
                body_start = start
                while body_start < len(value) and value[body_start] in " \t\r\n":
                    body_start += 1
                try:
                    _, end = json.JSONDecoder().raw_decode(value, body_start)
                except (ValueError, RecursionError):
                    self.invalid_call = True
                    return
                terminator = end
                while terminator < len(value) and value[terminator] in " \t\r\n":
                    terminator += 1
                if not value.startswith(_END, terminator):
                    self.invalid_call = True
                    return
                self.calls.append((self.name, value[body_start:end]))
                self.name, self.mode = "", _END
                start = terminator + len(_END)
                continue
            marker = _MARKER.search(value, start)
            while (
                marker is not None
                and marker.group() in (_TOOL_TEXT, _TOOL_ERROR)
                and self.mode != _MODEL
                and (self.mode not in _BOUNDARIES or marker.start() != start)
            ):
                marker = _MARKER.search(value, marker.end())
            end = marker.start() if marker else len(value)
            if marker is None and not final:
                if self.mode == _MODEL:
                    # A malformed unbounded header must never become visible.
                    if len(value) - start > 64 + _MAX_MARKER:
                        self.awaiting_final = True
                    else:
                        self.tail = value[start:]
                    return
                for length in range(min(_MAX_MARKER - 1, end - start), 0, -1):
                    if any(frame.startswith(value[end - length : end]) for frame in _FRAMES):
                        end -= length
                        break
                self.tail = value[end:]
            body = value[start:end]
            if self.mode == _MODEL:
                self.name = body
            elif self.mode == _THINKING:
                if body:
                    self.thinking(body)
            elif self.mode != _TOOL_ERROR and body:
                self.content(body)
            if marker is None:
                if self.mode in _BOUNDARIES and body:
                    self.mode = _TEXT
                break
            self.mode = marker.group()
            if self.mode != _TOOL:
                self.name = ""
            start = marker.end()
        if final and self.mode == _TOOL:
            self.invalid_call = True


def parse_response(text):
    content, thinking = [], []
    frames = OutputFrames(content.append, thinking.append)
    frames.parse(text)
    message = {"role": "assistant", "content": "".join(content)}
    if thinking:
        message["reasoning_content"] = "".join(thinking)
    return message, frames.calls, frames.invalid_call


class ResponseStream:
    def __init__(self, content, thinking):
        self.deliver = {"content": content, "reasoning_content": thinking}
        self.emitted = dict.fromkeys(self.deliver, 0)
        self.frames = OutputFrames(
            lambda text: self._emit("content", text),
            lambda text: self._emit("reasoning_content", text),
        )

    def _emit(self, field, text):
        if text:
            self.deliver[field](text)
            self.emitted[field] += len(text)

    def feed(self, text):
        self.frames.feed(text)

    def finish(self, message):
        for field, count in self.emitted.items():
            self._emit(field, message.get(field, "")[count:])
