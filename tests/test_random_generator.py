import matplotlib.pyplot as plt
import pytest

from cluster_import import from_csv
from ts_generator import ProbilityLaw, ThermalCluster, ThermalDataGenerator


def test_geometric_law(output_directory):
    generator = ThermalDataGenerator(days_per_year=1)

    A = [0]
    B = [0]

    law = ProbilityLaw.GEOMETRIC
    volatility = 1

    generator.prepare_outage_duration_constant(law, volatility, A, B, [10])

    expec = 0
    nb_values = 45
    values = [0] * nb_values
    N = 1000000
    N_inv = 1 / N
    for _ in range(N):
        value = generator.duration_generator(law, volatility, A[0], B[0], 10)
        assert value >= 1
        expec += value

        if value < nb_values:
            values[value] += N_inv

    expec /= N
    assert expec == pytest.approx(10, abs = 0.1)

    plt.plot(range(nb_values), values)
    plt.savefig(output_directory / "geometric_law_distrib.png")
    plt.clf()

def test_uniform_law(output_directory):
    generator = ThermalDataGenerator(days_per_year=1)

    A = [0]
    B = [0]

    law = ProbilityLaw.UNIFORM
    volatility = 1

    generator.prepare_outage_duration_constant(law, volatility, A, B, [10])

    expec = 0
    nb_values = 45
    values = [0] * nb_values
    N = 1000000
    N_inv = 1 / N
    for _ in range(N):
        value = generator.duration_generator(law, volatility, A[0], B[0], 10)
        assert value >= 1
        expec += value

        if value < nb_values:
            values[value] += N_inv

    expec /= N
    assert expec == pytest.approx(10, abs = 0.1)

    plt.plot(range(nb_values), values)
    plt.savefig(output_directory / "uniform_law_distrib.png")
    plt.clf()