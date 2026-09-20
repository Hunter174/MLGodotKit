extends GutTest

func make_controller() -> MovementControllerNode2D:
	var controller := MovementControllerNode2D.new()
	controller.arrival_radius = 1.0
	controller.set_limits(0.0, 100.0)
	return controller

func test_moves_toward_target():
	var controller := make_controller()
	controller.set_target(Vector2(10.0, 0.0))

	var velocity := controller.update(Vector2.ZERO, 0.1)

	assert_eq(velocity, Vector2(20.0, 0.0))
	controller.free()

func test_stops_inside_arrival_radius():
	var controller := make_controller()
	controller.set_target(Vector2(0.5, 0.0))

	assert_eq(controller.update(Vector2.ZERO, 0.1), Vector2.ZERO)
	controller.free()

func test_applies_speed_limit():
	var controller := make_controller()
	controller.set_target(Vector2(100.0, 0.0))
	controller.set_limits(0.0, 5.0)

	assert_eq(controller.update(Vector2.ZERO, 0.1), Vector2(5.0, 0.0))
	controller.free()
