"""Transfer chopped vegetables with a full-size folding bamboo preparation board.

The two rigid bamboo leaves bend about a passive silicone centre hinge. Food
moves by gravity and native contact; there are no food or board pose writes.
"""
from __future__ import annotations
import math
import pathlib
import sys
from collections import deque
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from yam_arena.task_file import TABLE_TOP,Box
import partkit as pk
import meshkit as mk
SCENE='station'
TOP=TABLE_TOP[SCENE]
INSTRUCTION='Transfer all the chopped carrot pieces from the folding bamboo board into the saucepan without losing any, and leave the board on its preparation rack and both grippers clear.'
SCORE_OVER='end'
ORIGIN=np.array([.46,0.,TOP+.115])
POT=np.array([.565,0.,TOP])
WOOD=(.65,.43,.22)
BOARD=pk.part('board',ORIGIN,[((0.,-.053,0.),(.260,.104,.010),WOOD)],.160,friction=.55,
 hinges=(pk.Hinge('leaf',(0.,0.,0.),boxes=(((0.,.053,0.),(.260,.104,.010),WOOD),),mass=.160,axis='X',lower=0.,upper=2.6,stiffness=.025,damping=.015,target=0.),),
 meta={'self_collide':True,'contact_offset':.0005})
# Open wire rack: 100 mm legs, top rods at 110 mm. The front 70 mm
# overhang gives unobstructed access to the board's actual top/bottom faces.
RACK=pk.part('rack',(.45,0.,TOP),[],.5,fixed=True,cylinders=tuple(
 [((x,y,.0525),.003,.105,(.55,.55,.55),'Z') for x in [-.04,.04] for y in [-.085,.085]]+
 [((x,0.,.107),.003,.19,(.55,.55,.55),'Y') for x in [-.04,0.,.04]]+
 [((0.,y,.107),.003,.10,(.55,.55,.55),'X') for y in [-.085,.085]]))
# Full-size small saucepan, 170 mm interior diameter, 95 mm rim height.
PAN=pk.part('saucepan',POT,[((.135,0.,.070),(.13,.025,.015),(.15,.15,.15))],.55,friction=.6)
mk.attach(PAN,[mk.Shape(mk.cup(.089,.095,.004,floor=.005),color=(.52,.54,.57),collider='decomposition',max_hulls=48)])
PIECES=[f'carrot_{i}' for i in range(8)]
OBJECTS=[BOARD,RACK,PAN]+[Box(name=n,pos=tuple(map(float,ORIGIN+np.array([x,y,.012]))),size=(.012,.012,.014),mass=.0018,color=(.95,.35,.05)) for n,(x,y) in zip(PIECES,[(x,y) for x in [-.06,.03] for y in [-.075,-.025,.025,.075]])]
_ST={}
def rx(a):
 c,s=math.cos(a),math.sin(a);return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
def ry(a):
 c,s=math.cos(a),math.sin(a);return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
@pk.on_reset
def reset_state(environment=None):
 _ST.clear();_ST.update(step=0,last_step=-1,last=(0.,False),history=deque(maxlen=21),violation='',released=False)
reset_state()
@pk.on_step
def update_step(environment):
 _ST['step']+=1
 # These are scorer observations, never inputs to demo actions.
 pts=pk.finger_points(environment)
 board=pk._part_state(environment,BOARD)
 pan=pk._part_state(environment,PAN)
 _ST['released']=all(np.linalg.norm(np.asarray(p)-board.pos)>.19 and np.linalg.norm(np.asarray(p)-pan.pos)>.14 for p in pts)
FOOD_HALF=np.array([.006,.006,.007])
CONTACT_TOL=.001

def food_corners(state):
 """Eight real box corners, respecting the piece's measured orientation."""
 x,y,z,w=state.quat_xyzw
 R=pk._quat_matrix((w,x,y,z))
 offsets=np.array([(a,b,c) for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)])*FOOD_HALF
 return offsets@R.T+state.pos

def food_inside(state,pan):
 local=(food_corners(state)-pan.pos)@pan.rotation()
 return bool(np.all(np.linalg.norm(local[:,:2],axis=1)<=.085+CONTACT_TOL)
             and np.all(local[:,2]>=.005-CONTACT_TOL)
             and np.all(local[:,2]<=.095+CONTACT_TOL))

def evaluate(objects):
 board=objects['board'];pan=objects['saucepan']
 if np.linalg.norm(pan.pos[:2]-POT[:2])>.020 or abs(pan.pos[2]-POT[2])>.010:_ST['violation']='saucepan moved beyond limit'
 if pan.rotation()[2,2]<math.cos(math.radians(10)):_ST['violation']='saucepan tipped'
 if board.pos[2]<TOP+.025:_ST['violation']='board fell'
 inside=[]
 for n in PIECES:
  p=objects[n].pos
  if p[2]<TOP-.005:_ST['violation']='food fell from worktop'
  # A piece on the bench outside the pan is a permanent loss. Contact
  # transient over the rim is permitted; the pan floor is 5 mm above bench.
  if food_corners(objects[n])[:,2].min()<=TOP+.001 and not food_inside(objects[n],pan):_ST['violation']='food spilled onto worktop'
  inside.append(food_inside(objects[n],pan))
 rack=(np.linalg.norm(board.pos[:2]-ORIGIN[:2])<=.025 and abs(board.pos[2]-ORIGIN[2])<=.006 and abs(board.rotation()[2,2])>math.cos(math.radians(8)) and abs(board.hinges['leaf'])<math.radians(8))
 full=all(inside) and rack and _ST['released']
 points=np.vstack([food_corners(objects[n]) for n in PIECES]+[np.array([board.to_world(p) for p in [(0.,0.,0.),(.12,-.1,0.)]])]+[np.array([board.to_world(rx(board.hinges['leaf'])@np.array([.12,.1,0.]))])])
 _ST['history'].append((points,full))
 if _ST['violation']:return 0.,False
 hist=_ST['history'];still=len(hist)==21 and all(f for _,f in hist) and max(np.linalg.norm(p-hist[-1][0],axis=1).max() for p,_ in hist)<=.001
 if full:return (1.,True) if still else (.75,False)
 if not _ST['released']:return 0.,False
 n=sum(inside)
 return (.5 if n==8 else .25 if n>=4 else 0.),False
def once(objects):
 if _ST['last_step']!=_ST['step']:_ST['last_step']=_ST['step'];_ST['last']=evaluate(objects)
 return _ST['last']
def score(objects):return once(objects)[0]
def success(objects):return once(objects)[1]
def distance(objects):return sum(float(np.linalg.norm(objects[n].pos-POT)) for n in PIECES)
def demo():return []
