class_name LinalgTestUtils

static func mat(a: Array) -> Matrix:
	return Matrix.from_array(a)

static func approx_eq(a: Matrix, b: Matrix, tol := 1e-4) -> bool:
	if a.rows() != b.rows() or a.cols() != b.cols():
		return false

	for i in a.rows():
		for j in a.cols():
			if abs(a.get(i, j) - b.get(i, j)) > tol:
				return false
	return true
