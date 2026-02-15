extends GutTest

func test_ce_forward_and_backward():

	var loss := CELossNode.new()

	var prediction = [[0.1, 0.9]]
	var target     = [[0.0, 1.0]]

	var value = loss.forward(prediction, target)
	var grad  = loss.backward()

	var expected_loss = -log(0.9)

	assert_almost_eq(value, expected_loss, 0.0001)

	assert_almost_eq(grad[0][0], 0.1, 0.0001)
	assert_almost_eq(grad[0][1], -0.1, 0.0001)
