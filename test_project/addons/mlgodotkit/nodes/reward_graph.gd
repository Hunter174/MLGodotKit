extends Node2D
class_name RewardGraph

@export var max_points: int = 100
@export var graph_size: Vector2 = Vector2(500, 250)
@export var offset: Vector2 = Vector2(50, 50)
@export var show_moving_average: bool = true
@export var avg_window: int = 10

var rewards: Array = []

func add_reward(value: float) -> void:
	rewards.append(value)
	if rewards.size() > max_points:
		rewards.pop_front()
	queue_redraw()

func _draw() -> void:
	if rewards.size() < 2:
		return

	# --- 1. Determine drawing bounds ---
	var max_val = ceil(max(1.0, rewards.max()))
	var min_val = floor(min(-1.0, rewards.min()))
	var y_range = max_val - min_val
	if y_range == 0:
		y_range = 1.0

	var scale_x = graph_size.x / float(max(1, rewards.size() - 1))
	var scale_y = graph_size.y / y_range

	# --- 2. Draw axes ---
	var origin = offset + Vector2(0, graph_size.y)
	var x_axis_end = offset + Vector2(graph_size.x, graph_size.y)
	var y_axis_end = offset

	draw_line(origin, x_axis_end, Color.WHITE, 2.0)
	draw_line(origin, y_axis_end, Color.WHITE, 2.0)

	# --- 3. Draw Y-axis tick marks & labels ---
	var tick_count = 5
	for i in range(tick_count + 1):
		var y_val = lerp(min_val, max_val, float(i) / tick_count)
		var y = offset.y + graph_size.y - (y_val - min_val) * scale_y
		draw_line(Vector2(offset.x - 5, y), Vector2(offset.x, y), Color.WHITE, 1.0)
		draw_string(ThemeDB.fallback_font, Vector2(offset.x - 45, y + 5), str(round(y_val)), HORIZONTAL_ALIGNMENT_RIGHT)

	# --- 4. Draw X-axis tick marks (episode numbers) ---
	var step_x_ticks = max(1, rewards.size() / 5)
	for i in range(0, rewards.size(), step_x_ticks):
		var x = offset.x + i * scale_x
		draw_line(Vector2(x, offset.y + graph_size.y), Vector2(x, offset.y + graph_size.y + 5), Color.WHITE, 1.0)
		draw_string(ThemeDB.fallback_font, Vector2(x - 10, offset.y + graph_size.y + 20), str(i), HORIZONTAL_ALIGNMENT_CENTER)

	# --- 5. Draw reward curve ---
	var prev_point = offset + Vector2(0, graph_size.y - (rewards[0] - min_val) * scale_y)
	for i in range(1, rewards.size()):
		var x = offset.x + i * scale_x
		var y = offset.y + graph_size.y - (rewards[i] - min_val) * scale_y
		var point = Vector2(x, y)
		draw_line(prev_point, point, Color.CORNFLOWER_BLUE, 2.0)
		prev_point = point

	# --- 6. Optional moving average curve ---
	if show_moving_average and rewards.size() > avg_window:
		var smoothed: Array = []
		for i in range(rewards.size()):
			var start = max(0, i - avg_window)
			var subset = rewards.slice(start, i + 1)
			smoothed.append(subset.reduce(func(a,b): return a + b) / subset.size())

		var prev_avg = offset + Vector2(0, graph_size.y - (smoothed[0] - min_val) * scale_y)
		for i in range(1, smoothed.size()):
			var x = offset.x + i * scale_x
			var y = offset.y + graph_size.y - (smoothed[i] - min_val) * scale_y
			var point = Vector2(x, y)
			draw_line(prev_avg, point, Color(1, 0.5, 0), 2.0)  # orange moving average
			prev_avg = point

	# --- 7. Draw border ---
	draw_rect(Rect2(offset, graph_size), Color(0.5, 0.5, 0.5, 0.5), false, 2.0)

	# --- 8. Labels ---
	draw_string(ThemeDB.fallback_font, offset + Vector2(graph_size.x / 2 - 40, graph_size.y + 40), "Episode", HORIZONTAL_ALIGNMENT_CENTER)
	draw_string(ThemeDB.fallback_font, offset + Vector2(-40, -10), "Reward", HORIZONTAL_ALIGNMENT_LEFT)

func _on_rl_node_rewards_log_updated(new_rewards: Array) -> void:
	rewards = new_rewards.duplicate()
	queue_redraw()
