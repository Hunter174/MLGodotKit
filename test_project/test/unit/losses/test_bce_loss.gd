extends GutTest

func test_bce_forward_and_backward():

	var loss := BCELossNode.new()

	var prediction = [[0.8]]
	var target     = [[1.0]]

	var value = loss.forward(prediction, target)
	var grad  = loss.backward()

	var expected_loss = -log(0.8)

	assert_almost_eq(value, expected_loss, 0.0001)
	assert_almost_eq(grad[0][0], -0.2, 0.0001)
