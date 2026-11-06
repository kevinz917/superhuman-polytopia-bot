import numpy as np
from polytopia_bot.gpu.mapgen import Layout
from polytopia_bot.learning.splits import board_hashes


def test_split_symmetry_hash_includes_all_board_fields_and_seat_swap():
    size = 4
    terrain = np.arange(16).reshape(size, size)
    structure = np.zeros((size, size), dtype=int)
    resource = structure.copy()
    capitals = np.array([1, 10])
    original = Layout(terrain.flatten(), structure.flatten(), resource.flatten(), capitals)
    raw, symmetric = board_hashes(original, size)
    grid = np.arange(16).reshape(size, size)
    for rotation in range(4):
        for flip in (False, True):
            transform = lambda x: np.flip(np.rot90(x, rotation), axis=0) if flip else np.rot90(x, rotation)
            index = transform(grid).flatten()
            moved = np.array([np.flatnonzero(index == c)[0] for c in capitals])
            for caps in (moved, moved[::-1]):
                board = Layout(transform(terrain).flatten(), transform(structure).flatten(), transform(resource).flatten(), caps)
                assert board_hashes(board, size)[1] == symmetric
    changed = original._replace(resource=np.ones(16, dtype=int))
    assert board_hashes(changed, size)[1] != symmetric
