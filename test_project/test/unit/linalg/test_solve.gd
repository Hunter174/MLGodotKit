extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_solve_exact_system():
	var A = U.mat([[2, 1], [1, 3]])
	var b = U.mat([[1], [2]])

	var x = Linalg.solve(A, b)
	assert_true(x != null)

	var expected = U.mat([[0.2], [0.6]])
	assert_true(U.approx_eq(x, expected))
