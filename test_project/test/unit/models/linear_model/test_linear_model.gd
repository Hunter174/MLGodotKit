extends GutTest

func test_lr_learns_simple_linear_function():
	var lr := LinearModelNode.new()
	lr.set_learning_rate(0.01)
	lr.initialize(1)

	var X = [[1],[2],[3],[4],[5]]
	var y = [[8],[11],[14],[17],[20]]  # y = 3x + 5

	lr.train(X, y, 1000)

	for i in X.size():
		var pred = lr.predict(X[i])[0]
		assert_true(abs(pred[0] - y[i][0]) < 1.0)
		
	lr.free()
