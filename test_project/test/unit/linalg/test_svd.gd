extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_svd_reconstruction():
	var A = U.mat([[1, 0], [0, 2]])

	var svd = Linalg.svd(A)
	var Umat = svd["U"]
	var S = svd["S"]
	var V = svd["V"]

	var reconstructed = Umat.matmul(S).matmul(V.transpose())
	assert_true(U.approx_eq(reconstructed, A))
