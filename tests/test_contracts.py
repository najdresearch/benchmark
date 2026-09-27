import copy
import json
from pathlib import Path

import httpx
import pytest

from najd_benchmark.contracts import digest, load_pack, seal_bundle, validate_bundle
from najd_benchmark.support import parse_label, run

PACK = Path(__file__).parents[1] / 'examples/arabic-support-routing-v1'


def sample_bundle():
    manifest, pack, cases = load_pack(PACK)
    count = 0
    def endpoint(request):
        nonlocal count
        count += 1
        if count == 2:
            return httpx.Response(200, json={'choices': [{'message': {'content': 'billing'}}]})
        if count == 3:
            raise httpx.ReadTimeout('secret endpoint details must not be recorded')
        return httpx.Response(200, json={'model': 'fixture-v1', 'choices': [
            {'message': {'content': '{"label":"billing"}'}}]})
    return run(manifest, pack, cases, endpoint='http://localhost/v1', model='fixture',
               endpoint_kind='local', transport=httpx.MockTransport(endpoint))


def test_round_trip_and_accounting():
    bundle = sample_bundle()
    validate_bundle(bundle)
    assert bundle['metrics']['case_count'] == 8
    assert bundle['metrics']['correct'] == 1
    assert bundle['metrics']['accuracy'] == 1/8
    assert bundle['metrics']['invalid_outputs'] == 1
    assert bundle['metrics']['provider_errors'] == 1
    assert bundle['metrics']['grading_coverage'] == 7/8
    assert bundle['metrics']['cost_usd'] is None
    assert bundle['metrics']['tokens']['cases_with_missing_usage'] == 8
    assert 'secret endpoint' not in json.dumps(bundle)


def test_ids_hashes_and_metrics_are_enforced():
    bundle = sample_bundle()
    bad = copy.deepcopy(bundle)
    bad['records'][0]['output'] = 'tampered'
    with pytest.raises(ValueError):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad['records'][1]['case_id'] = bad['records'][0]['case_id']
    with pytest.raises(ValueError, match='IDs'):
        seal_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad['metrics']['accuracy'] = 1
    with pytest.raises(ValueError, match='Metrics'):
        seal_bundle(bad)


def test_origin_is_not_attestation():
    bundle = sample_bundle()
    bundle['run']['origin'] = 'arena_managed'
    seal_bundle(bundle)  # valid self-reported data never establishes managed execution
    assert bundle['run']['visibility'] == 'private'


def test_output_contract_and_canonical_json():
    assert parse_label('{"label":"billing","extra":true}', ['billing']) is None
    assert parse_label('["billing"]', ['billing']) is None
    assert digest({'a': 1.0, 'ب': 2}) == digest({'ب': 2, 'a': 1})
