from ci import hygiene


def test_every_gate_has_a_failing_test():
    assert hygiene.gate_tests() == []
