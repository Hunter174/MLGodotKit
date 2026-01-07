extends GutTest

func test_first_sample_is_passthrough():
	var f := LowPassFilter3D.new()
	var v := Vector3(10, 20, 30)
	assert_eq(f.filter(v, 0.016), v)

func test_smooths_step_input():
	var f := LowPassFilter3D.new()
	f.cutoff_hz = 1.5
	f.reset(Vector3.ZERO)

	var a := f.filter(Vector3(100, 0, 0), 0.1)
	var b := f.filter(Vector3(100, 0, 0), 0.1)

	assert_true(a.length() < 100)
	assert_true(b.length() > a.length())

func test_disabled_is_passthrough():
	var f := LowPassFilter3D.new()
	f.enabled = false

	var v := Vector3(5, 15, 25)
	assert_eq(f.filter(v, 0.016), v)
