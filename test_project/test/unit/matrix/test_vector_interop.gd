extends GutTest

func test_from_vector3_column():
	var v = Vector3(1,2,3)
	var M = Matrix.from_vector3(v, true)
	assert_eq(M.rows(), 3)
	assert_eq(M.cols(), 1)

func test_to_vector3():
	var M = Matrix.from_array([[1],[2],[3]])
	var v = M.to_vector3()
	assert_eq(v, Vector3(1,2,3))

func test_mul_vector3():
	var I = Matrix.identity(3)
	var v = Vector3(1,2,3)
	assert_eq(I.mul_vector3(v), v)
