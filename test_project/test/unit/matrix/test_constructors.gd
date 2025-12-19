extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_zeros():
	var A = Matrix.zeros(2, 3)
	assert_eq(A.rows(), 2)
	assert_eq(A.cols(), 3)
	assert_eq(A.norm(), 0.0)

func test_ones():
	var A = Matrix.ones(2, 2)
	assert_eq(A.get(0, 0), 1.0)
	assert_eq(A.get(1, 1), 1.0)

func test_identity():
	var I = Matrix.identity(3)
	assert_eq(I.get(0, 0), 1.0)
	assert_eq(I.get(1, 0), 0.0)
