extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_qr_reconstruction():
	var A = U.mat([[1, 2], [3, 4]])

	var qr = Linalg.qr(A)
	assert_true(qr.has("Q"))
	assert_true(qr.has("R"))

	var Q = qr["Q"]
	var R = qr["R"]

	var reconstructed = Q.matmul(R)
	assert_true(U.approx_eq(reconstructed, A))
