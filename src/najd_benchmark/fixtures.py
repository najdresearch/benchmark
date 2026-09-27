"""Bounded, in-memory file tools for public fixture tasks (fixture-tools-v1)."""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import PurePosixPath
from typing import Any

PROTOCOL = "fixture-tools-v1"
MAX_TURNS = 12
MAX_BYTES = 64_000


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Invalid fixture path')
    if str(path) != value or value == '.':
        raise ValueError('Fixture path must be canonical')
    return value


async def run_fixture(
    prompt: str, files: dict[str, str],
    invoke: Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]],
    *, system_prompt: str | None = None,
) -> dict[str, Any]:
    """invoke returns content, model and usage; no native tool API is required.

    The model never sees gold answers. Its only filesystem is the supplied mapping.
    Inputs are immutable. Writes are restricted to a separate output mapping.
    """
    started = time.monotonic()
    inputs = {safe_path(k): v for k, v in files.items()}
    if sum(len(v.encode()) for v in inputs.values()) > MAX_BYTES:
        raise ValueError('Fixture exceeds size budget')
    outputs: dict[str, str] = {}
    instruction = (
        'You have a private virtual filesystem. Read the supplied files to solve the task. '
        'Respond with exactly one JSON object each turn: '
        '{"action":"read","path":"file"}, '
        '{"action":"write","path":"answer.txt","content":"text"}, or '
        '{"action":"finish","answer":"your final response"}. '
        'Source files are read-only. You can only read listed source files or your own outputs. '
        f'You have at most {MAX_TURNS} turns. Source paths: {json.dumps(sorted(inputs))}'
    )
    messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': prompt}]
    if system_prompt:
        messages[0]['content'] += '\n' + system_prompt
    transcript = []
    usage: dict[str, int] = {}
    model = ''
    for _ in range(MAX_TURNS):
        response = await invoke(messages)
        content = str(response.get('content') or '')
        model = response.get('model', model)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[key] = usage.get(key, 0) + int(response.get('usage', {}).get(key, 0) or 0)
        messages.append({'role': 'assistant', 'content': content})
        try:
            action = json.loads(content)
            if not isinstance(action, dict):
                raise ValueError('Expected one JSON object')
            operation = action.get('action')
            if operation == 'finish':
                if not isinstance(action.get('answer'), str):
                    raise ValueError('answer must be text')
                transcript.append({'response': content, 'result': 'finished',
                                   'model': model, 'usage': response.get('usage', {})})
                return {'output': action['answer'], 'files': outputs, 'transcript': transcript,
                        'usage': usage, 'model': model, 'protocol': PROTOCOL, 'status': 'complete',
                        'elapsed_seconds': time.monotonic() - started}
            path = safe_path(action.get('path', ''))
            if operation == 'read':
                result = inputs[path] if path in inputs else outputs[path]
            elif operation == 'write':
                if path in inputs:
                    raise ValueError('Source files are read-only')
                text = action.get('content')
                if not isinstance(text, str):
                    raise ValueError('content must be text')
                proposed = {**outputs, path: text}
                if sum(len(v.encode()) for v in proposed.values()) > MAX_BYTES:
                    raise ValueError('Output size budget exceeded')
                outputs[path] = text
                result = 'written'
            else:
                raise ValueError('Unknown action')
        except (ValueError, KeyError, TypeError) as exc:
            result = f'Error: {exc}'
        transcript.append({'response': content, 'result': result,
                           'model': model, 'usage': response.get('usage', {})})
        messages.append({'role': 'user', 'content': json.dumps({'tool_result': result},
                                                              ensure_ascii=False)})
    return {'output': '', 'files': outputs, 'transcript': transcript, 'usage': usage,
            'model': model, 'protocol': PROTOCOL, 'status': 'turn_limit',
            'elapsed_seconds': time.monotonic() - started}


def artifact_grade(expected: dict, result: dict) -> dict | None:
    """Fail missing execution/artifacts before semantic judging; None means judge next."""
    if result.get('protocol') != PROTOCOL:
        raise ValueError('Missing fixture execution evidence')
    reason = None
    if result.get('status') != 'complete':
        reason = 'Fixture turn limit reached'
    for path, required in expected.get('file_contains', {}).items():
        if required not in result.get('files', {}).get(path, ''):
            reason = f'Required output artifact missing or incorrect: {path}'
    if reason:
        return {'method': PROTOCOL, 'score': 0.0, 'passed': False, 'reason': reason}
    return None
