extends GutTest
const U = preload("res://test/unit/linalg/linalg_test_utils.gd")

func test_eig_symmetric_matrix():
	var A = U.mat([[2, 0], [0, 3]])

	var eig = Linalg.eig(A)
	assert_true(eig.has("values"))
	assert_true(eig.has("vectors"))

	var values = eig["values"]
	assert_eq(values.get(0, 0), 2.0)
	assert_eq(values.get(1, 0), 3.0)
