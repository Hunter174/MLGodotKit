extends Node2D

@export var projectile_speed := 600.0
@export var fire_interval := 0.5
@export var projectile_radius := 6.0
@export var projectile_lifetime := 2.0

@onready var target = $"../Target"

var fire_timer := 0.0
var active_projectiles := []

func _process(delta):
	fire_timer += delta
	if fire_timer >= fire_interval:
		fire_timer = 0.0
		fire_at_target()

	update_projectiles(delta)
	queue_redraw()

# ----------------------------
# Classical analytic leading
# ----------------------------
func fire_at_target():
	if target == null:
		return

	var shooter_pos = global_position
	var target_pos = target.global_position
	var target_vel = target.velocity

	var aim_angle = compute_lead_angle(
		shooter_pos,
		target_pos,
		target_vel,
		projectile_speed
	)

	spawn_projectile(aim_angle)

func compute_lead_angle(shooter_pos, target_pos, target_vel, speed):
	var rel_pos = target_pos - shooter_pos
	var rel_vel = target_vel

	var a = rel_vel.dot(rel_vel) - speed * speed
	var b = 2.0 * rel_pos.dot(rel_vel)
	var c = rel_pos.dot(rel_pos)

	var disc = b * b - 4.0 * a * c

	# Fallback: aim directly
	if disc < 0.0 or abs(a) < 0.0001:
		return rel_pos.angle()

	var sqrt_disc = sqrt(disc)
	var t1 = (-b - sqrt_disc) / (2.0 * a)
	var t2 = (-b + sqrt_disc) / (2.0 * a)

	var t = min(t1, t2)
	if t < 0.0:
		t = max(t1, t2)

	if t < 0.0:
		return rel_pos.angle()

	var aim_point = target_pos + target_vel * t
	return (aim_point - shooter_pos).angle()

# ----------------------------
# Projectile handling (visual only)
# ----------------------------
func spawn_projectile(angle):
	var projectile = {
		"pos": global_position,
		"vel": Vector2.RIGHT.rotated(angle) * projectile_speed,
		"time_left": projectile_lifetime
	}
	active_projectiles.append(projectile)

func update_projectiles(delta):
	for p in active_projectiles.duplicate():
		p.pos += p.vel * delta
		p.time_left -= delta

		if p.time_left <= 0.0:
			active_projectiles.erase(p)

# ----------------------------
# Rendering
# ----------------------------
func _draw():
	for p in active_projectiles:
		draw_circle(to_local(p.pos), projectile_radius, Color.RED)
