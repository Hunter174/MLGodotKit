extends Node2D
class_name RewardGraph

@export var max_points: int = 100
@export var graph_size: Vector2 = Vector2(400, 200)
@export var offset: Vector2 = Vector2(800, 50)

var rewards: Array = []

func add_reward(value: float) -> void:
	rewards.append(value)
	if rewards.size() > max_points:
		rewards.pop_front()
	queue_redraw()

func _draw() -> void:
	if rewards.size() < 2:
		return
	
	var max_val = max(1.0, rewards.max())
	var min_val = min(-1.0, rewards.min())

	var scale_x = graph_size.x / float(max_points)
	var scale_y = graph_size.y / (max_val - min_val)

	var prev_point = Vector2.ZERO
	for i in range(rewards.size()):
		var x = offset.x + i * scale_x
		var y = offset.y + graph_size.y - (rewards[i] - min_val) * scale_y
		var point = Vector2(x, y)
		if i > 0:
			draw_line(prev_point, point, Color.CORNFLOWER_BLUE, 2.0)
		prev_point = point

	# Draw border
	draw_rect(Rect2(offset, graph_size), Color(0.5, 0.5, 0.5, 0.5), false, 8.0)
