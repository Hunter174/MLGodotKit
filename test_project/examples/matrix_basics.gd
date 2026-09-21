extends Control

func _ready() -> void:
	var left := Matrix.from_array([
		[1.0, 2.0],
		[3.0, 4.0],
	])
	var right := Matrix.identity(2)
	var product := left.matmul(right)
	_show_result("Matrix basics", [
		"left = %s" % [left.to_array()],
		"right = identity(2)",
		"left × right = %s" % [product.to_array()],
		"shape = %d × %d" % [product.rows(), product.cols()],
		"det(left) = %.2f" % left.det(),
	])

func _show_result(title: String, lines: Array) -> void:
	var label := Label.new()
	label.position = Vector2(32, 32)
	label.add_theme_font_size_override("font_size", 20)
	label.text = title + "\n\n" + "\n".join(lines)
	add_child(label)
