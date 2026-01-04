extends Node2D
class_name Navigator2D

# ─────────────────────────────────────────────
# Navigation behaviors
# ─────────────────────────────────────────────
enum Behavior {
	SEEK,
	FLEE,
	LOS_SEEK,
	MAINTAIN_RANGE,
	PATROL,
	ORBIT
}

@export var behavior: Behavior = Behavior.SEEK

# ─────────────────────────────────────────────
# Obstacle sensing
# ─────────────────────────────────────────────
@export var obstacle_layers := 1
@export var ray_length := 120.0
@export var probe_angles := [-90, -60, -30, 0, 30, 60, 90]
@export var avoidance_weight := 1.2

# ─────────────────────────────────────────────
# Behavior parameters
# ─────────────────────────────────────────────
@export var desired_range := 96.0
@export var range_tolerance := 12.0
@export var orbit_direction := 1 # 1 = CCW, -1 = CW

@export var patrol_points: Array[Vector2] = []
var _patrol_index := 0

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
func compute_direction(
	position: Vector2,
	velocity: Vector2,
	target: Vector2,
	delta: float
) -> Vector2:

	var intent_dir := _compute_behavior_direction(position, target)
	var avoid_dir := _compute_avoidance(position, intent_dir)

	var dir := intent_dir + avoid_dir * avoidance_weight
	if dir.length() < 0.001:
		return intent_dir

	return dir.normalized()

# ─────────────────────────────────────────────
# Behavior resolution
# ─────────────────────────────────────────────
func _compute_behavior_direction(pos: Vector2, target: Vector2) -> Vector2:
	match behavior:
		Behavior.SEEK:
			return (target - pos).normalized()

		Behavior.FLEE:
			return (pos - target).normalized()

		Behavior.LOS_SEEK:
			if _has_los(pos, target):
				return Vector2.ZERO
			return (target - pos).normalized()

		Behavior.MAINTAIN_RANGE:
			var d := pos.distance_to(target)
			if abs(d - desired_range) <= range_tolerance:
				return Vector2.ZERO
			if d > desired_range:
				return (target - pos).normalized()
			return (pos - target).normalized()

		Behavior.ORBIT:
			var radial := (target - pos).normalized()
			return Vector2(-radial.y, radial.x) * orbit_direction

		Behavior.PATROL:
			if patrol_points.is_empty():
				return Vector2.ZERO
			var p := patrol_points[_patrol_index]
			if pos.distance_to(p) < 16.0:
				_patrol_index = (_patrol_index + 1) % patrol_points.size()
			return (p - pos).normalized()

	return Vector2.ZERO

# ─────────────────────────────────────────────
# Obstacle avoidance (steering-style)
# ─────────────────────────────────────────────
func _compute_avoidance(pos: Vector2, forward: Vector2) -> Vector2:
	var force := Vector2.ZERO

	for angle in probe_angles:
		var dir := forward.rotated(deg_to_rad(angle))
		var hit := _raycast(pos, pos + dir * ray_length)

		if not hit.is_empty():
			var dist := pos.distance_to(hit.position)
			var weight = 1.0 - clamp(dist / ray_length, 0.0, 1.0)
			force -= dir * weight

	return force.normalized()

# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────
func _space():
	return get_world_2d().direct_space_state

func _raycast(from: Vector2, to: Vector2) -> Dictionary:
	var q := PhysicsRayQueryParameters2D.create(from, to)
	q.exclude = []
	q.collision_mask = obstacle_layers
	return _space().intersect_ray(q)

func _has_los(from: Vector2, to: Vector2) -> bool:
	return _raycast(from, to).is_empty()
