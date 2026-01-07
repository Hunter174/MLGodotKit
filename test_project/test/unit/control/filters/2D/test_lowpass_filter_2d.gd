extends GutTest

func test_first_sample_is_passthrough():
	var f := LowPassFilter2D.new()
	var v := Vector2(50, 20)
	assert_eq(f.filter(v, 0.016), v)

func test_smooths_step_input():
	var f := LowPassFilter2D.new()
	f.cutoff_hz = 2.0
	f.reset(Vector2.ZERO)

	var a := f.filter(Vector2(100, 0), 0.1)
	var b := f.filter(Vector2(100, 0), 0.1)

	assert_true(a.length() < 100)
	assert_true(b.length() > a.length())

func test_high_cutoff_behaves_like_passthrough():
	var f := LowPassFilter2D.new()
	f.cutoff_hz = 1000.0

	var v := Vector2(40, 60)
	var out := f.filter(v, 0.016)

	assert_almost_eq(out.x, v.x, 0.01)
	assert_almost_eq(out.y, v.y, 0.01)

func test_disabled_is_passthrough():
	var f := LowPassFilter2D.new()
	f.enabled = false

	var v := Vector2(10, 90)
	assert_eq(f.filter(v, 0.016), v)
