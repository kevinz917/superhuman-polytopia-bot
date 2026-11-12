"""The Plato T3 release files are the ones the source manifest describes (hashes, weight header, board bank)."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'docs/plato-t3-source-manifest.json').read_text())


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_every_exported_file_matches_the_manifest():
    for row in MANIFEST['files']:
        assert sha256(row['path']) == row['exported_sha256'], row['path']
        assert (row['source_sha256'] != row['exported_sha256']) == row['modified_for_public_release'], row['path']
    for path in MANIFEST['authored_or_adapted']:
        assert (ROOT / path).is_file(), path


def test_weights_are_the_t3_policy():
    meta = json.loads((ROOT / 'weights/plato-t3/plato.json').read_text())
    assert (meta['stage'], meta['architecture'], meta['obs_version']) == ('update-96070', 'transformer-v3-p2-h4', 6)
    assert sha256('weights/plato-t3/plato.bin') == 'f1c61ed405decdf3473cbcb70aa166098f97863351179b46f1147cd90366f149'
    data = (ROOT / 'weights/plato-t3/plato.bin').read_bytes()
    assert data[:4] == b'PTB3'
    (count,) = struct.unpack_from('<I', data, 4)
    offset, elements = 8, 0
    for _ in range(count):
        (rank,) = struct.unpack_from('<I', data, offset)
        size = 1
        for dim in struct.unpack_from(f'<{rank}I', data, offset + 4):
            size *= dim
        elements += size
        offset += 4 + 4 * rank
    assert count == len(meta['leaves']) == 106
    assert elements == meta['parameter_elements'] == 3243386
    assert offset + 4 * elements == len(data)


def test_engine_binary_and_board_bank():
    assert sha256('plato-t3/runtime/engine_wasm.wasm') == 'd5525b0cd2f774065cffd927c7ef5ce577813fe2e71378800887aa9737e7ffde'
    for size in (11, 14, 16, 18, 20):
        boards = json.loads((ROOT / f'plato-t3/runtime/bank-{size}.json').read_text())
        mirrors = [b for b in boards if b['tribes'] == [4, 4]]
        assert all(b['size'] == size and len(b['terrain']) == size * size for b in boards)
        assert len(mirrors) == 24 and {b['map'] for b in mirrors} == {'dryland', 'lakes', 'pangea', 'continents', 'archipelago', 'waterworld'}
