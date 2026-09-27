import asyncio
import json

import pytest

from najd_benchmark.fixtures import MAX_TURNS, PROTOCOL, artifact_grade, run_fixture


def run(actions, files=None):
    actions = iter(actions)
    async def invoke(messages):
        # Answers must never leak into the model context.
        assert 'file_contains' not in str(messages)
        return {'content': json.dumps(next(actions)), 'model': 'fake',
                'usage': {'total_tokens': 10}}
    return asyncio.run(run_fixture('Read contacts.csv and write answer.txt',
                                  files or {'contacts.csv': 'city,owner\nRiyadh,Lulwa'}, invoke))


def test_read_write_finish_with_evidence():
    result = run([{'action': 'read', 'path': 'contacts.csv'},
                  {'action': 'write', 'path': 'answer.txt', 'content': 'Lulwa'},
                  {'action': 'finish', 'answer': 'Lulwa'}])
    assert result['files'] == {'answer.txt': 'Lulwa'}
    assert result['usage']['total_tokens'] == 30
    assert 'Riyadh' in result['transcript'][0]['result']
    assert artifact_grade({'file_contains': {'answer.txt': 'Lulwa'}}, result) is None


def test_claimed_write_is_not_execution():
    result = run([{'action': 'finish', 'answer': 'I wrote Lulwa to answer.txt'}])
    assert artifact_grade({'file_contains': {'answer.txt': 'Lulwa'}}, result)['score'] == 0


@pytest.mark.parametrize('action', [
    {'action': 'read', 'path': '/etc/passwd'},
    {'action': 'write', 'path': '../secret', 'content': 'x'},
    {'action': 'write', 'path': 'contacts.csv', 'content': 'tamper'},
    {'action': 'write', 'path': 'huge', 'content': 'x' * 64001},
    {'action': 'shell', 'path': 'contacts.csv'},
])
def test_tools_are_bounded(action):
    result = run([action, {'action': 'finish', 'answer': 'done'}])
    assert result['transcript'][0]['result'].startswith('Error:')
    assert not result['files']


def test_turn_limit():
    result = run([{'action': 'read', 'path': 'contacts.csv'}] * MAX_TURNS)
    assert result['status'] == 'turn_limit'
    assert artifact_grade({}, result)['score'] == 0
    assert result['protocol'] == PROTOCOL


def test_corrected_absher_key():
    from najd_benchmark.absher_corrections import grade
    assert grade({'answer': 'أ'}, 'أ')['score'] == 1
    assert grade({'answer': 'أ'}, 'أ أو ب')['score'] == 0
    assert grade({'answer': 'أ'}, 'ب')['score'] == 0
