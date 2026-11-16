---
name: stark-user-test
description: Run an explicitly requested desktop user simulation of Stark with OpenWork, OpenCode Desktop, and Pi, including a real project and independent acceptance checks.
---

# Stark user test

Run only when the user calls this test. Default to all three clients; honor an
explicit client subset. This is a live desktop workflow operated by Codex through
computer use. The fixture helper does not operate the apps. Never schedule it or
substitute direct API, mock, or standalone OpenCode CLI results for desktop results.

Resolve this skill directory's real path (including a home-directory symlink)
and use `git -C REAL_SKILL_DIRECTORY rev-parse --show-toplevel` to locate its
Stark checkout. Read that checkout's `AGENTS.md`
and [desktop procedure](../../../docs/USER_SIMULATION.md) before starting. Run the
helper from that checkout using `uv run --no-project --python 3.13.13 python
scripts/stark_user_test.py`. Record its returned absolute run directory. Run one
client at a time against one loaded model. Delegate independent helper/review
work when useful; serialize desktop control and inference.

Use actual app UI and Pi's Terminal interface to connect, select Stark, submit
the authored briefs, handle tool approvals, stop work, and recover. Keep the
same fresh conversation through build, revise and recover. Give the client only
its own project folder, sample input and the current brief. Do not implement or
repair `ledger.py` yourself, expose the oracle or reference solution, or patch
client/provider code to get a pass. The helper's `brief` command returns authored
test instructions; these are not captured user conversations.

Observe tool execution and Stark's request activity. A written file or a claimed
completion does not prove the client completed its task. Independently run
`verify` after each stage and record UI checks only after seeing the named
behavior. Feed back only failing check IDs for up to two repair turns per stage.
Retain failures in the aggregate history. Allow at most ten minutes per task
turn and two minutes for cancellation or connection recovery; user-specified
budgets take precedence. On timeout, cancel in the client, confirm Stark idle,
record the failed/blocked step, and continue other independent clients when safe.

Use the current running model unless the user specifies another. Never silently
substitute a provider, disable image support or MTP, install dependencies, rotate
the API key, restart an unrelated engine, or delete existing sessions. If no
suitable model is running, or a client is missing, record a blocked result and
continue available work. Request necessary missing choices without guessing a
large download. Pause for OS credential prompts or an approval the user must
personally provide; continue once the outcome is observed.

Keep credentials in the existing Connections/hidden-prompt flow. Do not read or
print Keychain values, discovery bearer tokens, environment dumps or client
databases. Do not export/save chat transcripts, prompts, responses, tool payloads,
raw subprocess output, exception dumps or screenshots. Only synthetic project
files and aggregate helper records stay in the private ignored run directory.

Finish by closing only the test sessions, restoring changed working-folder and
connection choices to their observed initial state, then generating `report`.
Leave the private project/evidence available for review. A blocked/missing step
is incomplete. Report actual outcomes and the report path; never call the three
app workflow passed from model-free helper tests. Distinguish operator-observed
UI checks, independent project checks, and unknown model artifact hashes.
