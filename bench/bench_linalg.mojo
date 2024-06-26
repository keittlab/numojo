from benchmark import *
from linalg.static_matrix import StaticMat
from stochasticity.splitmix import SplitMix

from python import Python


fn bench_mat_mat_mult[rows: Int, cols: Int]() raises -> Report:
    var rng = SplitMix()
    var x = StaticMat[rows, cols](0)
    var y = StaticMat[cols, rows](0)
    for i in range(rows):
        for j in range(cols):
            x[i, j] = rng.next()
            y[j, i] = rng.next()

    fn doit() capturing:
        var z = x @ y
        keep(z)

    return run[doit]()


fn bench_mat_mat_mult_np[rows: Int, cols: Int]() raises -> Report:
    var np = Python.import_module("numpy")
    var rng = SplitMix()
    var x = np.zeros((rows, cols))
    var y = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            x[i, j] = rng.next()
            y[i, j] = rng.next()

    fn doit() capturing:
        try:
            var z = np.dot(x, y)
        except:
            pass

    return run[doit]()


fn main() raises:
    bench_mat_mat_mult[3, 3]().print("ns")
    # bench_mat_mat_mult_np[3, 3]().print("ns")
