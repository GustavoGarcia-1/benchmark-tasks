# Bounce a table-tennis ball into a cup

Bounce the white table-tennis ball once on the board into the red cup, leaving the ball inside the cup and both hands clear.

This familiar tabletop challenge tests release timing and rebound trajectory control. A free 40 mm ball must reach a separate cup after a real contact bounce; it is not a tethered cup-and-ball toy or paddle keep-ups. No alcohol or liquid is involved. The native PhysX sphere is accelerated only by the robot, gravity and contact. The scorer reads the resulting trajectory; it never changes it.

Files: task.py defines the scene, events and scorer; meshkit.py creates the tapered cup and native sphere collider; partkit.py supplies finger positions for release checks; demo.py is an open-loop demonstration. No external assets are used.

```sh
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "Bounce the white table-tennis ball once on the board into the red cup, leaving the ball inside the cup and both hands clear." --scorer task_score --max-steps 1600 --store-frames --log-dir dream
python demo.py task.py demo --max-steps 1600
DEMO_VARIANT=held python demo.py task.py held --max-steps 500 --no-record
DEMO_VARIANT=straightdrop python demo.py task.py straightdrop --max-steps 500 --no-record
```

One fixed layout, object_jitter=0, no random seeds. A trial lasts at most 80 seconds. The demonstration uses fixed scene dimensions and its own arm commands; ball state appears only in diagnostic logging. Its release sweep starts at x120/y-150 mm, with the TCP 240 mm above the board and leaned 15 degrees forward. It advances 25 mm per 50 ms control step, opening on the fifth step and withdrawing after one further step. The cup centre is aligned with the measured entry line to avoid adding off-centre orbital momentum. The cup placement was checked against a measured native free-flight arc; no trajectory is imposed on the ball.

Scoring: 1.0 for exactly one released board bounce followed by containment, hands 55 mm clear of ball centre, and a 20-step/1 s centre-position window within 1 mm. 0.75 for the same geometry without the complete stability window. 0.25 is retained credit for making one valid released board bounce. Otherwise 0. A second board bounce, falling below the table, or any hand approaching within 55 mm of the ball centre after release permanently gives 0; regrasping and placing cannot score. Containment requires centre height 20.5-95 mm above the board and normal clearance of 20 mm from the exact tapered inner cup wall. A bounce requires an actual native ball/board contact episode, upward-facing normal (absolute z at least 0.95), contact position within the board footprint and within 2 mm of the board top, outside the 30 mm-radius disk covered by the cup base. The Isaac Sim 6 experimental ContactSensor reads native raw contacts at physics substeps, so a fast ball penetrating during one substep cannot be mistaken for a different surface from sparse control-step position samples. Ball/cup and ball/table contacts do not count. Verify the corresponding real trial using slow-motion video, ruler measurements and the final second of video. There is no restriction on which hand acts.

## Real trial

Use a dry 40 mm ABS table-tennis ball (2.7 g), a red polypropylene cup matching 95 mm rim diameter, 60 mm base diameter and 115 mm height (measure with calipers; replace a mismatched cup), and a smooth 360 x 260 x 6 mm laminated board. These ordinary household items and removable tape cost below $30. Tape the board flat to the table and tape the cup base to it. The board centre is x290/y-150 mm, the cup centre x378/y-180.34 mm, and the ball centre starts x200/y-150 mm, 27 mm above the table. There is no liquid or sharp edge.

Before use, drop the ball vertically with its centre 220 mm above the board. Accept a peak rebound centre between 175 and 190 mm, measured on slow-motion video against a vertical ruler; choose a different laminated board/ball if outside this band. This calibrates the effective ball/board restitution target of 0.90. Do not claim that an arbitrary table surface has that value. The ITTF ball leaflet specifies the 40 mm/2.7 g ball and a standard steel-block drop test; this task's actual board is separately calibrated: https://documents.ittf.sport/sites/default/files/public/2020-10/2020_BoD_documents_2.pdf . Cup and board dimensions are explicit trial specifications, not a claimed manufacturer's product drawing.

Reset by retrieving the ball, checking for dents and loose tape, and replacing it on the marked start. Measure the cup and check the rebound before each batch. A person can check every score from the overhead and side video: count board bounces, check the final ball is wholly inside the cup, compare hand clearance against a 55 mm ruler, and inspect the last second for visible motion.

Fidelity: native spherical collision with 2.7 g mass and explicit hollow thin-shell inertia, 2mr²/3 = 7.2e-7 kg m² about each axis. The closed visual mesh supplies appearance, not solid-ball inertia. The cup is rigid and the board fixed. Air drag, shell deformation and rolling resistance are omitted; restitution represents the calibrated pair. Natural settling is slow: a captured ball is observed until the unchanged 1 s/1 mm stability test passes or the 1,600-step limit ends. No velocity or pose setters act on the ball after reset. The native demonstration reaches 1.0 in 1,468 steps. The held-ball and straight-drop negative controls each score 0.0 after 500 steps, after their full scripted actions and observation periods. The final policy evaluation is recorded below. Diagnostic object-state reads only write logs; they never choose robot actions.

Evaluation: exactly one Dream-1 trial, score 0.0, run limit 1,600 steps. Native demonstration score 1.0 at 1,468 steps; held and straight-drop controls 0.0 at 500 steps each.
