extends GutTest

func test_rows_cols():
	var A = Matrix.from_array([[1,2,3],[4,5,6]])
	assert_eq(A.rows(), 2)
	assert_eq(A.cols(), 3)

func test_get_set():
	var A = Matrix.zeros(2,2)
	A.set(1, 0, 5)
	assert_eq(A.get(1, 0), 5.0)
