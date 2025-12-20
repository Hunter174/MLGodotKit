extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_inverse_identity():
	var A = Matrix.from_array([[4,7],[2,6]])
	var Ainv = A.inverse()
	var I = A.matmul(Ainv)
	assert_true(U.approx_eq(I, Matrix.identity(2)))

func test_det():
	var A = Matrix.from_array([[1,2],[3,4]])
	assert_true(abs(A.det() + 2.0) < 1e-5)

func test_trace():
	var A = Matrix.from_array([[1,2],[3,4]])
	assert_eq(A.trace(), 5.0)
