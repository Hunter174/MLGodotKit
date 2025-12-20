extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_pinv_moore_penrose_identity():
	var A = U.mat([[1, 2], [3, 4]])
	var Ap = Linalg.pinv(A)

	var reconstructed = A.matmul(Ap).matmul(A)
	assert_true(U.approx_eq(reconstructed, A))
