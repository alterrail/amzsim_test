# 2D perception example

A standalone Python animation of a car following the middle of a cone track,
with a forward-facing **120° lidar field of view** (±60°). No ROS is needed.

```bash
python3 -m pip install -r examples/perception_2d/requirements.txt
python3 examples/perception_2d/simulate.py
python3 examples/perception_2d/simulate.py path/to/cones.csv
```

The included oval is runnable immediately. To use a repository track:

```bash
python3 examples/perception_2d/simulate.py src/planners/adaptive_velocity_profile/test/input/FSG/cones.csv
```

CSV input uses the repository's `tag,x,y` format, with coordinates in metres:

```csv
tag,x,y
blue,0,2
blue,5,2
yellow,0,-2
yellow,5,-2
```

Cones **must be ordered along each boundary**; blue and yellow rows can be
interleaved. Closed tracks require at least three cones of each color. For an
open track (including the short CSV above), use `--open-track`; two per side
suffice. Other tags are displayed and sensed but do not define the middle line.
Unordered maps or uncolored maps need boundary labeling and ordering first.

Both boundaries are sampled uniformly by arc length, aligned in direction and
(for closed tracks) starting phase, and averaged. The car moves along that
middle line at constant speed, starting near the first blue cone. This assumes
simple, corresponding track boundaries; complex junctions or very dissimilar
boundary lengths may require a supplied/hand-corrected track instead.

Every scan independently detects each cone inside the FOV and range with
`p(d) = exp(-d / decay_distance)`. Defaults: 30 m range, 20 m decay distance,
5 m/s speed, 10 Hz scans, and one lap. At 0, 10, 20, and 30 m the probabilities
are 100%, 61%, 37%, and 22%. Outside the FOV/range, probability is zero.
`--seed` makes runs repeatable. Gray rings show eligible cones; green rings show
successful detections in the current scan. The blue wedge shows the lidar view.

```bash
python3 examples/perception_2d/simulate.py --speed 8 --range 40 --decay-distance 15 --hz 10 --seed 42
python3 examples/perception_2d/simulate.py --headless --duration 10 --save /tmp/perception.png --log /tmp/detections.csv
python3 examples/perception_2d/simulate.py --headless --duration 5 --save /tmp/perception.gif
python3 -m unittest discover -s examples/perception_2d -p 'test_*.py'
```

PNG output shows the final scan; GIF output shows the full animation. Logs
include all cones for every scan, their eligibility, probability and detection
result, together with the car pose. Longer GIFs and runs use more memory because
frames are precomputed for reproducibility.

This illustrates perception availability rather than reproducing the production
point-cloud detector. The probability curve is illustrative, not calibrated to
recorded data. There is no occlusion, measurement noise, false positives, color
classification, or SLAM. Driving uses the full ground-truth track, independently
of missed detections, with ideal path following and no vehicle dynamics.
