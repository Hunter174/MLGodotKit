extends GutTest

func test_copy_is_deep():
	var A = Matrix.from_array([[1,2],[3,4]])
	var B = A.copy()
	B.set(0,0,99)
	assert_eq(A.get(0,0), 1.0)

func test_equals():
	var A = Matrix.from_array([[1,2],[3,4]])
	var B = Matrix.from_array([[1.000001,2],[3,4]])
	assert_true(A.equals(B, 1e-4))

func test_info():
	var A = Matrix.identity(3)
	var info = A.info()
	assert_eq(info["rows"], 3)
	assert_eq(info["cols"], 3)
