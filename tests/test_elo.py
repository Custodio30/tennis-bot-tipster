
from tennistips.elo import expected_score
def test_expected_score_symmetry():
    a = expected_score(1500, 1500)
    b = expected_score(1600, 1600)
    assert abs(a - 0.5) < 1e-9
    assert abs(b - 0.5) < 1e-9
