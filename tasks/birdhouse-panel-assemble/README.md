# Assemble a plywood birdhouse

Assemble the wooden birdhouse with all four walls and both roof panels fully fitted to its base, and leave it standing freely.

This is a seven-panel, full-size craft birdhouse. The base is initially on the work mat; the four walls and two pitched roof panels are loose in presentation racks. No panel is attached, teleported or snapped into position. This package supplies its own fabrication geometry: it is not an exact replica of the Greenleaf kit. Tab-and-slot birdhouse kits are sold as family craft projects, e.g. https://shop.greenleafdollhouses.com/swiss-inn-birdhouse/ .

The physical difficulty is closing a three-dimensional enclosure: each wall must register with two base slots, and each roof panel must engage both gables while resting on the roof slopes. The roof slots prevent a panel simply sliding off the 45-degree slope. This differs from an open slotted partition by closing across multiple already-installed members. 

## Files and runs

`task.py` defines geometry and scoring; `meshkit.py` and `partkit.py` spawn meshes and expose native arm fingertips; `demo.py` uses fixed waypoints with arm-joint feedback only; `clearance.py` measures arm clearance for logging; `export_patterns.py` generates the supplied full-scale SVG cut profiles and dimension lists in `fabrication/`. No third-party object assets are used.

```bash
python demo.py task.py demo_out --max-steps 2400
DEMO_VARIANT=no_roof python demo.py task.py no_roof --max-steps 2400 --no-record
DEMO_VARIANT=offset python demo.py task.py offset --max-steps 2400 --no-record
DEMO_VARIANT=hold python -c 'import demo; demo.run("task.py", "held", 1900, record=False, cam_width=80, cam_height=44)' 
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "Assemble the wooden birdhouse with all four walls and both roof panels fully fitted to its base, and leave it standing freely." --scorer task_score --max-steps 2400 --store-frames --log-dir dream
```

Fixed layout, object_jitter=0; no randomized seed. Limit 2,400 steps at 20 Hz, two simulated minutes. The scripted demonstration reaches 1.0 in 1,958 steps, with all seven panels released and stable and no recorded violation. Negative scores: omitted roofs 0.5 (1,210 steps), misplaced rear wall 0.0 (2,074 steps), held final roof 0.25 (1,900 steps). The held-roof test uses 80 x 44 camera images and no recording; physical geometry, control rate and scoring are unchanged. All panels finish within alignment tolerance, but the right roof is still gripped. Dream-1 scored 0.0 in one evaluation, reaching the 2,400-step limit. No additional Dream-1 trials were run.

## Real trial and geometry

Fabricate the base from 18 mm plywood and the other six panels from 6 mm plywood using the supplied SVGs at 1:1, checking the labelled dimensions in dimensions.json. Base 190 x 190 mm. Side walls 136 mm long, 107 mm high above their lower shoulders, with two 12 mm-deep bottom tabs, tapering from 16 x 6 mm at the shoulder to 12 x 2 mm at the tip. Front and rear are 144 mm wide, 110 mm to the eaves and 182 mm to the gable apex, with a pair of bottom tabs and two roof-retention tabs. The front has a 42 mm entrance. Roof panels are 248 x 113 mm with two 8 x 18 mm through-slots. Four wall planes are 144 mm apart. The complete enclosure is approximately 200 mm high, excluding small roof-retention projections. This is a tabletop craft assembly test, not a certified outdoor nesting box; no wildlife-use claim is made.

Use a router bit no larger than 3 mm for slot corners, or laser-cut the profiles. Routed slots are 8 mm for 6 mm panels, and 18 mm along each 16 mm tab. Sand sharp edges, verify actual thickness, and finish with dry clear varnish; do not put dust or wet paint on robot fingers. Sand both faces and sides of every bottom tab to the linear taper specified above. The tapered lead-ins must guide the joint physically; there are no attraction forces. These are clearance fits, not friction-press fits. Actual plywood density and contact friction vary: 600 kg/m³ and friction 0.65 are modelling estimates. There are no unspecified springs or force thresholds. Check that every dry joint enters by hand without hammering and that the roof tabs stop downslope sliding after release. If actual slot fit or friction differs materially, record the measured values and adjust the model rather than silently substituting a different rig.

The base is loose at x=280 mm, y=0 on a marked mat, between eight 10 mm-high bench locator blocks with 1 mm lateral clearance. These blocks locate only the base; they do not support the walls or attach the house. The base can lift vertically. Side-wall shoulders initially stand 14 mm above the table at (140,+/-320) mm. The front gable starts at (360,370), shoulder height 14 mm, plane along y. The rear gable starts at (390,-300), shoulder height 75 mm, plane along x, on a 150 x 26 x 61 mm shelf. Side-wall planes also run along x. Cheek pairs are 120 mm apart, with 9 mm-wide channels and 22 mm-long contact footprints, built from screwed hardwood scraps. Roof panels start at (490,+/-530,170) mm, sloping +/-45 degrees about x, on four blocks whose heights match each panel underside, plus an eave lip preventing downslope slipping in the rack. Exact support dimensions, positions and panel orientations are supplied in fabrication/rack_layout.json and generated in task.py. Positions are relative to the arm-base midpoint, z relative to the tabletop. Racks are fastened to the bench and are presentation aids only. All house parts are loose.

Budget estimate, not a supplier quote: plywood offcuts $15–25, rack offcuts and screws $10, local laser/router cutting $30–60, sandpaper/finish $10, total approximately $65–105 excluding reusable measuring tools. A builder with a jigsaw/drill can cut the same profiles. Buy or fabricate at the specified full scale, not a miniature. Verify the completed house holds together when both hands are clear; an unsupported leaning panel does not count.

## Scoring and checks

Judge current final state, not retained assembly milestones. A panel is seated when its body origin is within 3 mm and its orientation within 2 degrees of its assembled pose relative to the actual base. A placed panel earns credit only with every gripper fingertip at least 30 mm away from its bounding box. For the physical trial use a steel rule, a 3 mm feeler gauge, a square and an angle gauge to check base shoulders, wall plane spacing and roof slope; inspect that all tabs are through their corresponding slots. Side-view and top-view video document release and stability.

0.25: both side walls fully seated and released. 0.5: all four walls fully seated and released. 0.75: all walls and both roofs seated and all panels released, but the one-second stability check is incomplete. 1.0: the base bottom is within 1 mm of the tabletop, its tilt is at most 2 degrees, and all those conditions hold for one second, with four witness points per panel moving no more than 1 mm. Both success() and score() use this same gate. Zero permanently if any part falls off the bench, the base moves over 20 mm from its starting mark, or it tilts over 15 degrees. Otherwise zero. A held roof can never earn full credit. An omitted roof cannot earn above 0.5.

## Review criteria and limitations

Importance: recognizable family craft assembly, filed under Societal relatability; no workforce-size claim. Construct validity: real clearances, slots, gravity, rigid bodies and contact carry assembly; no scripted attachment. Scoring: physical, released alignment and one-second stillness, permanent violations. Clarity: one end-state instruction, fixed layout, supplied dimensioned cut geometry. Feasibility: affordable reproducible plywood rig; the complete GPU demonstration reaches 1.0. All three negative variants remain below full credit. The single Dream-1 evaluation scored 0.0 at the 2,400-step limit.
