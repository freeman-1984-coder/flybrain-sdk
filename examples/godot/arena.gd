extends RefCounted
# Engine-owned world. Same declared 20 ms rules as Python DodgeArena.
var rng: int = 7
var frame: int = 0
var x: float = 0.5
var blocks: Array = []
var collisions: int = 0
var passed: int = 0

func _init() -> void:
	spawn()

func spawn() -> void:
	rng = (1664525 * rng + 1013904223) % 4294967296
	blocks.append({"x": 0.15 + 0.7 * float(rng) / 4294967296.0, "y": 0.0})

func observe() -> Dictionary:
	var left: float = 0.0
	var right: float = 0.0
	for b in blocks:
		var dx: float = b.x - x
		var intensity: float = clampf((b.y - 0.1) / 0.65, 0, 1) * maxf(0, 1 - absf(dx) / 0.6)
		if dx <= 0: left = maxf(left, intensity)
		if dx >= 0: right = maxf(right, intensity)
	return {"danger_left": left, "danger_right": right, "player_x": x, "collisions": collisions, "passed": passed}

func apply(action: Dictionary) -> Dictionary:
	var previous: float = x
	x = clampf(x + clampf(action.steer, -1, 1) * 0.02, 0, 1)
	var remaining: Array = []
	for b in blocks:
		var y: float = b.y + 0.8 * 0.02
		if y >= 1:
			if absf(b.x - x) < 0.13: collisions += 1
			else: passed += 1
		else: remaining.append({"x": b.x, "y": y})
	blocks = remaining
	frame += 1
	if frame % 45 == 0: spawn()
	return {"steer": (x - previous) / 0.02}

func snapshot() -> Dictionary:
	return {"type": "dodge-arena-v1", "seed": 7, "rng": rng, "frame": frame, "x": x, "blocks": blocks.duplicate(true), "collisions": collisions, "passed": passed}

static func from_snapshot(data: Dictionary):
	if data.get("type") != "dodge-arena-v1" or data.get("seed") != 7:
		return null
	for key in ["rng", "frame", "collisions", "passed"]:
		var number = data.get(key)
		if not (number is float or number is int) or not is_finite(number) or number < 0 or floor(number) != number or number > 4294967295:
			return null
	if not data.get("x") is float or not is_finite(data.x) or data.x < 0 or data.x > 1:
		return null
	if not data.get("blocks") is Array or data.blocks.size() > 10000:
		return null
	for b in data.blocks:
		if not b is Dictionary: return null
		for key in ["x", "y"]:
			var number = b.get(key)
			if not (number is float or number is int) or not is_finite(number) or number < 0 or number > 1:
				return null
	var restored = load("res://arena.gd").new()
	restored.rng = int(data.rng)
	restored.frame = int(data.frame)
	restored.collisions = int(data.collisions)
	restored.passed = int(data.passed)
	restored.x = data.x
	restored.blocks = data.blocks.duplicate(true)
	return restored
