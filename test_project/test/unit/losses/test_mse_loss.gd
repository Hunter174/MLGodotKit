extends GutTest

func test_mse_forward_and_backward():

	var loss := MSELossNode.new()

	var prediction = [[0.8]]
	var target     = [[1.0]]

	var value = loss.forward(prediction, target)
	var grad  = loss.backward()

	assert_almost_eq(value, 0.04, 0.0001)

	assert_almost_eq(grad[0][0], -0.4, 0.0001)
