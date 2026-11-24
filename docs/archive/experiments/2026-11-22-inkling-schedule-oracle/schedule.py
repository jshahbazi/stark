"""Worker-shaped capture cases for the pinned Inkling 2 Mini capture runner.

On the source path the generation worker ingests an N-token text prompt as
bulk commands of at most 2,048 rows over the first N - 4 tokens, one three-row
tail command and a singleton seed step at position N - 1. The pinned capture
runner prefills its explicit chunk list and then evaluates four singleton
rows starting at Start. With Start = N - 1 and the bulk and tail commands as
chunks, row 0 is the generation seed and rows 1 to 3 are the following
teacher rows, exactly as in the archived stock-schedule 4K capture
(Start 4095, chunks [2048, 2044, 3]).

This module is pure Python and touches no model, tokenizer or GPU.
"""

SOURCE_ROWS = 2048
SOURCE_SUFFIX = 4
TEACHER_ROWS = 3
PREFIX_BOUNDARIES = (4096, 8192)
VERIFY_OUTPUTS = 96
MAX_VERIFY_ROWS = 8

# Case id -> (prompt tokens N, context, capture with the pinned runner).
# Short prompts exercise the short-prompt selector at the floor (69), one
# above it (73), two 64-row query tiles (130) and the old floor minus one
# (511). N = 68 is a self-comparison control below the floor. The long cases
# need the context-workspace selector: 4350 verifies across 4,352, 8188
# verifies across 8,192, 8800 places the bulk tail and every teacher row past
# 8,192, and 16000 is the 16K-class prompt with seven complete bulk commands.
CASES = {
    "s68": (68, 4352, False),
    "s69": (69, 4352, True),
    "s73": (73, 4352, True),
    "s130": (130, 4352, True),
    "s511": (511, 4352, True),
    # Remainder commands below the packed minimum (10 rows) and below the
    # source-kernel minimum (50 rows) inside a source-scoped request.
    "r4110": (4110, 4352, True),
    "r4150": (4150, 4352, True),
    "l4350": (4350, 9216, True),
    "l8188": (8188, 9216, True),
    "l8800": (8800, 9216, True),
    "l16000": (16000, 16384, True),
}


def source_chunks(prompt, rows=SOURCE_ROWS):
    """Bulk and tail command lengths before the seed for an N-token prompt."""
    if prompt <= SOURCE_SUFFIX or rows <= 0:
        raise ValueError("prompt must exceed the four-token source suffix")
    bulk = prompt - SOURCE_SUFFIX
    chunks = [rows] * (bulk // rows)
    if bulk % rows:
        chunks.append(bulk % rows)
    chunks.append(SOURCE_SUFFIX - 1)
    return chunks


def replay_boundaries(prompt, rows=SOURCE_ROWS):
    """Bulk command ends that the Stark checks replay from a snapshot."""
    ends, offset = [], 0
    for length in source_chunks(prompt, rows)[:-1]:
        offset += length
        ends.append(offset)
    worker = (prompt - SOURCE_SUFFIX) // rows * rows
    return [end for end in ends if end > 0 and (end in PREFIX_BOUNDARIES or end == worker)]


def request_fields(prompt):
    """Capture runner fields other than the private prompt text."""
    return {"Start": prompt - 1, "Chunks": source_chunks(prompt), "FullTarget": True}


def check_case(prompt, context):
    """Raise unless every Stark check fits the context."""
    if prompt + TEACHER_ROWS > context:
        raise ValueError("teacher rows exceed the context")
    if prompt + VERIFY_OUTPUTS + MAX_VERIFY_ROWS > context:
        raise ValueError("verifier outputs exceed the context")
