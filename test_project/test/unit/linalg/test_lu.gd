extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_lu_reconstruction():
	var A = U.mat([[4, 3], [6, 3]])

	var lu = Linalg.lu(A)
	var L = lu["L"]
	var Umat = lu["U"]
	var P = lu["P"]

	var reconstructed = P.matmul(L.matmul(Umat))
	assert_true(U.approx_eq(reconstructed, A))
