# Folding-board vegetable transfer

Transfer all the chopped carrot pieces from the folding bamboo board into the saucepan without losing any, and leave the board on its preparation rack and both grippers clear.

Everyday cooking involves carrying prepared ingredients into a pan. This carrier has two full-size bamboo leaves joined by a silicone hinge. They can form a chute. The goal permits any successful transfer technique; no fold milestone or angle is scored. Compared with fixed cutting-board scraper transfer (TK-KXZ89H), the moving carrier must retain the loose pieces during pickup, orientation and return.

## Physical specification and real trial

The reference board is Joseph Joseph Chop2Pot Bamboo small, SKU60111: 260 x210 x10 mm,320 g. The [manufacturer catalogue, page74](https://doc.formadore.com/catalog/joseph-joseph/Joseph_Joseph_2022.pdf) shows its centre silicone hinge and folding use. [Retailer specification](https://business.currys.co.uk/catalogue/domestic-appliances/small-kitchen-appliances/small-cooking-appliances/joseph-joseph-chop2pot-small-bamboo-chopping-board/N238955W) supplies mass and dimensions. Availability is not established; a reproducible substitute uses two bamboo leaves each260 x104 x10mm, with 2mm centre gap, bonded to a flexible food-safe silicone strip along the 260mm seam. Trim total mass to320g. Check hinge opens flat and bends to60degrees under a light hand force, without a mechanical catch. The simulated hinge's0.025 Nm/rad restoring stiffness is an estimate, not a manufacturer force claim.

Build the preparation rack from 6mm stainless rod: four105mm legs at x=+-40mm,y=+-85mm, three190mm crossbars at x=-40,0,40mm,z=107mm and two100mm rails at y=+-85mm,z=107mm. Weld or mechanically clamp rods rigidly and add nonslip feet; finished rod crowns110mm above the bench. Secure the rack to the bench with clamps outside the board and arm paths; its simulated base is fixed. Rods, feet and clamps total approximately500g. The near board edge overhangs70mm, giving access to its real top/bottom faces. Rack centre is450mm forward of the robot and on the midline; board centre460mm forward,115mm high. This elevated preparation support is part of the trial setup.

Use a 170 mm-inside-diameter saucepan 95 mm high, with a target mass of 550 g, centred 565 mm forward on the midline. Verify its cavity dimensions and mass before the trial. The model uses a 4 mm outer wall and 5 mm base as conservative collision geometry; these are not a claim that the real pan is made of solid stainless steel of those thicknesses. The interior containment surface and assigned mass define the simulation, with wall thickness an acknowledged approximation. Its130mm handle points forward. Eight dry carrot chunks12x12x14mm,1.8g each, start on the board in two rows at x=-60,+30mm and y=-75,-25,+25,+75mm. Use fresh pieces and a clean dry preparation board each trial; no powders or sharp implements contact the robot. Board, low-cost saucepan and welded/clamped rack can be obtained or built for less than a few hundred dollars. Use a ruler to mark all positions.

## Scoring and real checks

A piece counts inside when all eight transformed corners lie within the cylindrical pan interior: radius85mm, floor5mm and rim95mm above its underside, with1mm contact tolerance at these surfaces. This checks actual orientation and whole-piece containment rather than a conservative centre proxy. Photograph from above and inspect the rim to check containment. After every fingertip point is over 190 mm from the board hinge centre and over 140 mm from the saucepan base centre: four or more pieces inside gives0.25; all eight gives0.5. All eight plus board returned within25mm of its original centre and6mm of original height, flat on the rack (body normal within8degrees of vertical and leaves within8degrees of flat) gives0.75. That complete state held for1second, all tracked points moving at most1mm, gives1.0. A ruler and one-second video establish these levels.

Food lost on the worktop/floor, board hinge centre falling below25mm above the worktop, or saucepan translation over20mm, vertical movement over10mm or tilt over10degrees permanently gives0. No folding angle is required for food-transfer credit. Grippers still holding earn0. A successful alternative flat-board transfer remains valid.

## Files and execution

`task.py` is the task and scorer; `partkit.py` supplies native compound/hinged bodies; `meshkit.py` builds the hollow saucepan; `demo.py` uses only fixed geometry, timing and arm-joint feedback to generate actions; `clearance.py` supplies diagnostic robot geometry. No library meshes are used. Generated primitive geometry has no external asset licence requirement.

```sh
python demo.py task.py out/demo --max-steps 1200
DEMO_VARIANT=no_tip python demo.py task.py out/no_tip --max-steps 1200 --no-record
DEMO_VARIANT=held python demo.py task.py out/held --max-steps 1200 --no-record
DEMO_VARIANT=flat python demo.py task.py out/flat --max-steps 1200 --no-record
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "Transfer all the chopped carrot pieces from the folding bamboo board into the saucepan without losing any, and leave the board on its preparation rack and both grippers clear." --scorer task_score --max-steps 1200 --store-frames --log-dir out/dream
```

Limit1200steps, one minute at20Hz. Fixed layout,object_jitter=0,no random seeds. No-tip and held are negative controls. Flat is an alternative-technique experiment, not an automatically negative control. The current95mm front-edge-grasp Isaac demo reached1.0 in433steps (21.65seconds), with no scorer violation and worst reported IK error0.09mm for the left arm and0.13mm for the right. The recorded native-contact trial completes pickup, folding, pouring, return and release. Negative controls both scored 0.0: no-tip in 548 steps and held-board in 478 steps. Exactly one Dream-1 trial scored 0.0 at the 1,200-step limit (log adhoc_37d21d70). The run logged policy expiration/re-request messages; this is the sole recorded trial and was not repeated.

The recorded trial's minimum sampled robot-mesh arm-link separation was17.7mm. This is a sampled surface-distance diagnostic, not an exact minimum clearance measurement; mesh sampling can overestimate separation. The corresponding offline468-step path check, sampled every five steps, reported12.85mm minimum separation. Before a real trial, verify the complete path unloaded at low speed and low force, confirm actual wrist/forearm separation, and stop if interference occurs. Verify the rack clamps and gripper access before loading food.

The grippers close on the board's actual top/bottom faces at its front edge,95mm from the centre seam, with a35mm higher unloaded preapproach. At the initial left grasp, sampled front-contact finger geometry spans y=83.6-105.8mm against the leaf ending at105mm: approximately21.4mm overlaps the board and0.8mm extends beyond its edge; the right grasp is symmetric. The successful native-contact trial demonstrates retention through this partial overlap. The controller uses fixed timing and the arms' own joint readings; it does not steer from object poses or task joints.

Fidelity: rigid bamboo leaves approximate a commercial flexible silicone seam with a native passive hinge. Rigid carrot pieces, saucepan and rack use native collision and gravity. Friction and hinge torque remain estimates. No liquid, cutting, scripted attachment or food repositioning occurs.

Review assessment:

- Importance: transferring chopped vegetables into a pan is an ordinary meal-preparation step; this is a cooking task rather than a specialist equipment operation.
- Construct validity: two moving, hinged leaves carry eight loose rigid chunks through native contact and gravity. The measured demo completes that mechanism without attachment or food-pose scripts. Alternative transfer techniques remain valid; overlap with existing pouring tasks remains an acceptance risk.
- Scoring: whole-piece oriented corners check containment, gripper clearance gates credit, violations persist, and final food/board witness points must remain within1mm for one second.32 offline tests include tilted pieces, wall protrusions, near-pan spills, held states and moving pieces.
- Clarity: instruction, fixed layout, material sizes, physical reset, commands and per-level real checks are explicit. Rebuild dimensions specify the board, rack and pan; no hidden per-trial facts are used.
- Feasibility: the full-size board, buildable clamped rack and small saucepan form an affordable bench rig. Silicone-hinge torque and friction remain estimates, commercial board availability is unverified, and sampled arm clearances require physical verification before real execution.
