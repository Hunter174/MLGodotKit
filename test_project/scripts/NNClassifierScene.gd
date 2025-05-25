extends Node2D

@onready var point_area = $PointArea
@onready var label_button: Button = $LabelButton
@onready var train_button: Button = $TrainButton
@onready var loss_label: Label = $LossLabel

var point_map = Rect2(Vector2(100, 100), Vector2(400, 400))
var pred_map = Rect2(point_map.position + Vector2(500, 0), point_map.size)

var inputs = []
var targets = []
var loss_history: Array = []
var loss_points: PackedVector2Array = []


func _ready():
	_draw()  # Triggers _draw()

func _draw():
	draw_rect(point_map, Color.YELLOW, false, 2.0)
	draw_rect(pred_map, Color.CYAN, false, 2.0)

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if not point_map.has_point(event.position):
			print("Click outside point bounds:", event.position)
			return

		var class_label := 0
		if event.button_index == MOUSE_BUTTON_RIGHT:
			class_label = 1

		var input_pos = (event.position - point_map.position) / point_map.size
		inputs.append([input_pos.x, input_pos.y])
		targets.append([float(class_label)])

		var box = ColorRect.new()
		box.size = Vector2(10, 10)
		if class_label == 0:
			box.color = Color.BLUE
		else:
			box.color = Color.RED
		
		box.position = event.position - box.size / 2
		point_area.add_child(box)

func _train_and_visualize():
	if inputs.is_empty():
		print("No input data!")
		return

	var nn = NNNode.new()
	nn.set_learning_rate(0.1)
	nn.add_layer(2, 4, "relu")
	nn.add_layer(4, 4, "relu")
	nn.add_layer(4, 1, "sigmoid")

	loss_history.clear()

	for epoch in range(5000):
		var total_loss := 0.0
		for j in range(inputs.size()):
			var pred = nn.forward([inputs[j]])[0]
			var error = pred - targets[j][0]
			total_loss += error * error
			nn.backward([[2.0 * error]])
		loss_history.append(total_loss / inputs.size())
		
		if epoch % 100 == 0:
			await get_tree().process_frame

	# Create prediction map
	var tex_size = pred_map.size
	var img = Image.create(int(tex_size.x), int(tex_size.y), false, Image.FORMAT_RGB8)

	for x in range(img.get_width()):
		for y in range(img.get_height()):
			var fx = float(x) / img.get_width()
			var fy = float(y) / img.get_height()
			var out = nn.forward([[fx, fy]])[0]
			var brightness = clamp(out, 0.0, 1.0)
			var color = Color(brightness, brightness, brightness)
			img.set_pixel(x, y, color)

	img.generate_mipmaps()
	var tex = ImageTexture.create_from_image(img)

	var pred_sprite = Sprite2D.new()
	pred_sprite.texture = tex
	pred_sprite.position = pred_map.position + pred_map.size / 2  # center it
	add_child(pred_sprite)

	## Update loss label
	var final_loss = loss_history[-1]
	loss_label.text = "Final Loss: %.4f" % final_loss


func _plot_loss():
	var graph_origin = Vector2(pred_map.position.x, pred_map.position.y + pred_map.size.y + 20)
	var width := 400.0
	var height := 100.0

	var max_loss = max(0.001, loss_history.max())
	var x_scale = width / loss_history.size()
	var y_scale = height / max_loss

	loss_points.clear()

	for i in range(loss_history.size()):
		var x = graph_origin.x + i * x_scale
		var y = graph_origin.y + height - (loss_history[i] * y_scale)
		loss_points.append(Vector2(x, y))
	_draw()


func _on_train_button_pressed() -> void:
	_train_and_visualize()
