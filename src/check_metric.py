"""Unit check of apk/mapk against the course slide example (MAP@5: relevant A, B, C)."""
from common import apk, mapk

rel = {"A", "B", "C"}
x = apk(rel, ["A", "B", "D", "C", "E"])
y = apk(rel, ["D", "A", "E", "B", "C"])
print(f"Method X: {x:.3f} (slide 0.917) | Method Y: {y:.3f} (slide 0.533)")
assert round(x, 3) == 0.917 and round(y, 3) == 0.533
# edge cases: more relevant items than K -> divide by K; duplicates in a list are not double-counted
assert apk(set("ABCDEFG"), list("ABCDE")) == 1.0
assert apk({"A"}, ["A", "A", "B", "C", "D"]) == 1.0
assert mapk({1: ["A"], 2: ["B"]}, {1: ["A"], 2: ["X"]}) == 0.5
print("all metric checks passed")
