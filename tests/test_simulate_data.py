from ella.data.simulate_data import Bin, generate_r_bins


def test_generate_r_bins() -> None:
    assert [
        Bin(mid=0.125, left=0.0, right=0.25),
        Bin(mid=0.375, left=0.25, right=0.5),
        Bin(mid=0.625, left=0.5, right=0.75),
        Bin(mid=0.875, left=0.75, right=1.0),
    ] == generate_r_bins(4)
