extends GutTest

func test_first_sample_passthrough():
	var s := SlewLimiter3D.new()
	var v := Vector3(100, 0, 0)
	assert_eq(s.filter(v, 0.016), v)

func test_limits_rate_of_change():
	var s := SlewLimiter3D.new()
	s.max_delta_per_sec = 5.0
	s.reset(Vector3.ZERO)

	var out := s.filter(Vector3(100, 0, 0), 1.0)
	assert_almost_eq(out.length(), 5.0, 0.001)

func test_reset_sets_internal_state():
	var s := SlewLimiter3D.new()
	s.reset(Vector3(10, 20, 30))

	assert_eq(s.filter(Vector3(10, 20, 30), 0.016), Vector3(10, 20, 30))
