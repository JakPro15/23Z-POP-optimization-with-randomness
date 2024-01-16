from functions import check_bounds

def test_in_square():
    assert check_bounds(510) == 510

def test_out_of_square():
    assert check_bounds(514) == 510

def test_double_bounce():
    assert check_bounds(1537) == -511

def test_many_bounces():
    for l in range(100):
        for k in range(2, 10):
            assert check_bounds(512 + 1024 * k + l) == (512 - l) * (-1) ** k