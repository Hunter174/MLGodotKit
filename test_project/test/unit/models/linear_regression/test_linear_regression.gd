extends GutTest

func test_closed_form_lr_learns_simple_linear_function():
	var lr := LinearRegressionNode.new()

	# y = 3x + 5
	var X = [[1], [2], [3], [4], [5]]
	var y = [[8], [11], [14], [17], [20]]

	# Closed-form fit (single call)
	lr.fit(X, y)

	# Verify predictions
	print(lr.predict(X))
	for i in X.size():
		print(lr.predict([X[i]]))
		var pred = lr.predict([X[i]])[0][0]
		assert_true(
			abs(pred - y[i][0]) < 0.001,
			"Prediction %f deviates from expected %f" % [pred, y[i][0]]
		)

	lr.free()
