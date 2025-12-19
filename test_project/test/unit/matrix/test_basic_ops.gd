extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_transpose():
	var A = Matrix.from_array([[1,2,3]])
	var At = A.transpose()
	assert_eq(At.rows(), 3)
	assert_eq(At.cols(), 1)

func test_matmul():
	var A = Matrix.from_array([[1,2],[3,4]])
	var B = Matrix.from_array([[2],[1]])
	var C = A.matmul(B)
	assert_true(U.approx_eq(C, Matrix.from_array([[4],[10]])))
