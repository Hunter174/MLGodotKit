extends GutTest

func test_dtree_binary_threshold():
	var tree := DecisionTreeNode.new()
	tree.set_min_samples_split(2)
	tree.set_max_depth(10)

	var X_train = [
		[1.0], [2.0], [3.0], [4.0], [5.0],
		[6.0], [7.0], [8.0], [9.0], [10.0],
		[11.0], [12.0], [13.0], [14.0], [15.0]
	]
	var y_train = [
		0,0,0,0,0,
		0,0,0,0,0,
		1,1,1,1,1
	]

	tree.fit(X_train, y_train)

	var X_test = [[2.0], [5.0], [9.0], [11.0], [14.0]]
	var y_true = [0, 0, 0, 1, 1]
	var preds = tree.predict(X_test)

	for i in y_true.size():
		assert_eq(preds[i], y_true[i])
		
	tree.free()
