extends GutTest

func make_pid() -> PIDControllerNode:
	var pid := PIDControllerNode.new()
	pid.set_sample_time(0.1)   # 100 ms controller
	pid.set_limits(-10.0, 10.0)
	pid.set_tau(0.02)
	return pid


func test_zero_error_produces_zero_output():
	var pid = make_pid()
	pid.set_kp(1.0)
	pid.set_ki(1.0)
	pid.set_kd(1.0)

	var out = pid.update(5.0, 5.0)
	assert_almost_eq(out, 0.0, 0.0001)
	pid.free()


func test_proportional_only_response():
	var pid = make_pid()
	pid.set_kp(2.0)
	pid.set_ki(0.0)
	pid.set_kd(0.0)

	var out = pid.update(10.0, 7.0) # error = 3
	assert_almost_eq(out, 6.0, 0.0001)
	pid.free()


func test_integral_accumulates_error():
	var pid = make_pid()
	pid.set_kp(0.0)
	pid.set_ki(1.0)
	pid.set_kd(0.0)

	var out1 = pid.update(1.0, 0.0)
	var out2 = pid.update(1.0, 0.0)
	var out3 = pid.update(1.0, 0.0)

	assert_true(out2 > out1)
	assert_true(out3 > out2)
	pid.free()


func test_output_is_clamped():
	var pid = make_pid()
	pid.set_kp(100.0)
	pid.set_ki(0.0)
	pid.set_kd(0.0)
	pid.set_limits(-5.0, 5.0)

	var out = pid.update(1.0, 0.0)
	assert_eq(out, 5.0)
	pid.free()


func test_integrator_does_not_wind_up_when_saturated():
	var pid = make_pid()
	pid.set_kp(50.0)
	pid.set_ki(10.0)
	pid.set_kd(0.0)
	pid.set_limits(-5.0, 5.0)

	# Push into saturation
	for i in range(20):
		pid.update(1.0, 0.0)

	# Flip error sign
	var out = pid.update(0.0, 1.0)

	# Should recover quickly, not remain stuck
	assert_true(out < 0.0)
	pid.free()


func test_reset_clears_state():
	var pid = make_pid()
	pid.set_kp(1.0)
	pid.set_ki(1.0)

	pid.update(1.0, 0.0)
	pid.update(1.0, 0.0)
	pid.reset()

	var out = pid.update(1.0, 1.0)
	assert_almost_eq(out, 0.0, 0.0001)
	pid.free()


func test_no_derivative_kick_on_setpoint_change():
	var pid = make_pid()
	pid.set_kp(0.0)
	pid.set_ki(0.0)
	pid.set_kd(5.0)

	# Measurement constant
	pid.update(0.0, 0.0)
	var out = pid.update(10.0, 0.0)

	# Since derivative is on measurement, output should remain ~0
	assert_almost_eq(out, 0.0, 0.0001)
	pid.free()
