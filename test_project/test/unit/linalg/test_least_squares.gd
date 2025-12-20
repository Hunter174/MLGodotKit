extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_least_squares_overdetermined():
	var A = U.mat([[1, 1], [1, 2], [1, 3]])
	var b = U.mat([[1], [2], [2]])

	var x = Linalg.least_squares(A, b)
	assert_true(x != null)

	# Reference solution ≈ [0.67, 0.5]
	assert_true(abs(x.get(0, 0) - 0.67) < 0.05)
	assert_true(abs(x.get(1, 0) - 0.5) < 0.05)
