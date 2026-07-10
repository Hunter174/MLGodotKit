extends Node2D

# ----------------------------
# Tunables
# ----------------------------
@export var projectile_speed := 600.0
@export var projectile_radius := 6.0
@export var projectile_lifetime := 2.0
@export var fire_interval := 0.5

@onready var neural_network = $NeuralNetworkNode
@onready var target = $"../Target"

# ----------------------------
# Normalization constants
# ----------------------------
const MAX_POS := 800.0
const MAX_VEL := 400.0
const MAX_DIST := 1200.0
const MAX_SPEED := 1000.0

func normalize_pos(v):
	return clamp(v / MAX_POS, -1.0, 1.0)

func normalize_vel(v):
	return clamp(v / MAX_VEL, -1.0, 1.0)

func normalize_dist(v):
	return clamp(v / MAX_DIST, 0.0, 1.0)

func normalize_speed(v):
	return clamp(v / MAX_SPEED, 0.0, 1.0)

# ----------------------------
# Runtime state
# ----------------------------
var fire_timer := 0.0
var last_shot_data := {}
var active_projectiles := []   # array of dictionaries

# ----------------------------
# Initialization
# ----------------------------
func _ready():
	# Lets make sure the NN node was initalized correctly
	neural_network.model_summary()

# ----------------------------
# Main loop
# ----------------------------
func _process(delta):
	fire_timer += delta
	if fire_timer >= fire_interval:
		fire_timer = 0.0
		fire_at_target()

	update_projectiles(delta)

# ----------------------------
# Firing + inference
# ----------------------------
func fire_at_target():
	var to_target = target.global_position - global_position
	var dist = to_target.length()
	var base_angle = to_target.angle()
	var v = target.velocity

	var input = [
		normalize_pos(to_target.x),
		normalize_pos(to_target.y),
		normalize_vel(v.x),
		normalize_vel(v.y),
		normalize_speed(projectile_speed),
		normalize_dist(dist)
	]

	var output = neural_network.forward(input)
	var aim_offset = output[0][0] # Known bug will fix in a future update
	var final_angle = base_angle + aim_offset

	spawn_projectile(final_angle)

	last_shot_data = {
		"final_angle": final_angle,
		"target": target
	}

# ----------------------------
# Projectile data management
# ----------------------------
func spawn_projectile(angle):
	var projectile = {
		"pos": global_position,
		"vel": Vector2.RIGHT.rotated(angle) * projectile_speed,
		"time_left": projectile_lifetime
	}

	active_projectiles.append(projectile)
	queue_redraw()

func update_projectiles(delta):
	for p in active_projectiles.duplicate():
		p.pos += p.vel * delta
		p.time_left -= delta

		if p.time_left <= 0.0:
			active_projectiles.erase(p)
			on_projectile_resolved(false, p.pos)

	queue_redraw()

# ----------------------------
# Rendering
# ----------------------------
func _draw():
	for p in active_projectiles:
		draw_circle(to_local(p.pos), projectile_radius, Color.WHITE)

# ----------------------------
# Training
# ----------------------------
func on_projectile_resolved(hit, impact_position):
	if last_shot_data.is_empty():
		return

	var target = last_shot_data.target
	var fired_angle = last_shot_data.final_angle
	var ideal_angle = compute_intercept_angle(target)

	var angle_error = wrapf(ideal_angle - fired_angle, -PI, PI)
	neural_network.backward([angle_error])

	last_shot_data.clear()

# ----------------------------
# Physics-based ideal solution
# ----------------------------
func compute_intercept_angle(target):
	var rel_pos = target.global_position - global_position
	var rel_vel = target.velocity

	var t = rel_pos.length() / projectile_speed
	var future_pos = target.global_position + rel_vel * t

	return (future_pos - global_position).angle()
