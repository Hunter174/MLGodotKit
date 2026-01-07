extends GutTest

func test_first_sample_passthrough():
	var s := SlewLimiter2D.new()
	var v := Vector2(100, 0)
	assert_eq(s.filter(v, 0.016), v)

func test_limits_rate_of_change():
	var s := SlewLimiter2D.new()
	s.max_delta_per_sec = 10.0
	s.reset(Vector2.ZERO)

	var out := s.filter(Vector2(100, 0), 1.0)
	assert_almost_eq(out.length(), 10.0, 0.001)

func test_reset_sets_internal_state():
	var s := SlewLimiter2D.new()
	s.reset(Vector2(25, 25))

	assert_eq(s.filter(Vector2(25, 25), 0.016), Vector2(25, 25))

func test_disabled_is_passthrough():
	var s := SlewLimiter2D.new()
	s.enabled = false

	var v := Vector2(200, -40)
	assert_eq(s.filter(v, 0.016), v)
