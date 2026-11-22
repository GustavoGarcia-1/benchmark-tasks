"""Assemble a full-size seven-panel plywood craft birdhouse with physical tab joints.

The supplied cut patterns define the real trial; this is an authored kit, not an
unmeasured replica of a commercial product. No pose snapping or joint creation.
All dimensions are metres. Base 190 mm square, 18 mm thick; walls/roof 6 mm.
"""
from __future__ import annotations
import math
import pathlib
import sys
from collections import deque
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import meshkit as mk
from partkit import finger_points
from yam_arena.task_file import TABLE_TOP, Box

SCENE = 'station'
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Assemble the wooden birdhouse with all four walls and both roof panels fully fitted to its base, and leave it standing freely."
SCORE_OVER = 'end'
WOOD = (0.72, 0.53, 0.31)
ROOF = (0.48, 0.26, 0.16)
T = .006
BASE_T = .018
HALF = .072
EAVE = .110
RISE = .072
CX = .28
SLOT = .008
TAB_W = .016
TAB_D = .012
RELEASE = .030
STILL_STEPS = 21
_ST = {}

def rot_x(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[1.,0,0],[0,c,-s],[0,s,c]])

def quat(R):
    w,x,y,z=mk._matrix_quat_wxyz(np.asarray(R))
    return (x,y,z,w)

def rect(u0,v0,u1,v1):
    return [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]

def panel_mesh(poly, holes=(), thickness=T):
    if len(poly)>4 and min(v for _,v in poly)<0:
        # The two real plywood tabs are bevelled over their 12 mm length:
        # 16 x 6 mm at the shoulder, 12 x 2 mm at the tip. Closed touching
        # mesh components preserve the physical lead-in without scripted forces.
        outline=[]
        for i,a in enumerate(poly):
            b=poly[(i+1)%len(poly)]
            if a[1]>=0 and (not outline or a!=outline[-1]):outline.append(a)
            if (a[1]<0)!=(b[1]<0):
                f=-a[1]/(b[1]-a[1]);q=(a[0]+f*(b[0]-a[0]),0.)
                if not outline or q!=outline[-1]:outline.append(q)
        if outline[0]==outline[-1]:outline.pop()
        meshes=[mk.extrude(outline,thickness,holes=holes,z0=-thickness/2)]
        for u in (-.04,.04):
            m=mk.extrude(rect(u-.008,-TAB_D,u+.008,0),thickness,z0=-thickness/2)
            v=m.points.copy();f=-v[:,1]/TAB_D
            v[:,0]=u+(v[:,0]-u)*(1-.25*f)
            v[:,2]*=1-(2/3)*f
            meshes.append(mk.Mesh(v,m.faces))
        pts=[];faces=[];offset=0
        for m in meshes:pts.extend(m.points);faces.extend(m.faces+offset);offset+=len(m.points)
        pts=np.asarray(pts);faces=np.asarray(faces)
        xyz=pts[faces];centres=xyz[:,:,0].mean(axis=1)
        join=np.all(np.abs(xyz[:,:,1])<1e-10,axis=1)&((np.abs(centres+.04)<.00801)|(np.abs(centres-.04)<.00801))
        faces=faces[~join]
        unique,inverse=np.unique(np.round(pts,10),axis=0,return_inverse=True)
        return mk.Mesh(unique,inverse[faces])
    return mk.extrude(poly,thickness,holes=holes,z0=-thickness/2)

def tabs(poly):
    # Ordered lower edge with two downward tabs, then the remaining perimeter.
    return [(-HALF,0),(-.048,0),(-.046,-TAB_D),(-.034,-TAB_D),(-.032,0),
            (.032,0),(.034,-TAB_D),(.046,-TAB_D),(.048,0),(HALF,0)]+poly

BASE_HOLES=[]
for x in (-HALF,HALF):
    for y in (-.04,.04):BASE_HOLES.append(rect(x-SLOT/2,y-.009,x+SLOT/2,y+.009))
for y in (-HALF,HALF):
    for x in (-.04,.04):BASE_HOLES.append(rect(x-.009,y-SLOT/2,x+.009,y+SLOT/2))
BASE_POLY=rect(-.095,-.095,.095,.095)
SIDE_POLY=[(max(-.068,min(.068,u)),v) for u,v in tabs([(HALF,EAVE-T/2),(-HALF,EAVE-T/2)])]
# Each gable slope includes a tab perpendicular to the sloped roof. The tab
# penetrates a true through-slot in the roof and prevents sliding down the eave.
def gable_poly():
    # Trace right slope upwards, then left slope downwards.
    out=tabs([(HALF,EAVE)])
    for sign,vals in [(1,(.04+.008/math.sqrt(2),.04-.008/math.sqrt(2))),
                      (-1,(-.04+.008/math.sqrt(2),-.04-.008/math.sqrt(2)))]:
        if sign == -1:out.append((0,EAVE+RISE))
        for j,y in enumerate(vals):
            z=EAVE+RISE-abs(y)
            if j==0:out.extend([(y,z),(y+sign*TAB_D/math.sqrt(2),z+TAB_D/math.sqrt(2))])
            else:out.extend([(y+sign*TAB_D/math.sqrt(2),z+TAB_D/math.sqrt(2)),(y,z)])
    out.append((-HALF,EAVE))
    return out
GABLE_POLY=gable_poly()
HOLE=[(.021*math.cos(a),.075+.021*math.sin(a)) for a in np.linspace(0,2*math.pi,33)[:-1]]
ROOF_POLY=rect(-.124,-.0565,.124,.0565)
ROOF_HOLES=[rect(x-SLOT/2,-.009,x+SLOT/2,.009) for x in (-HALF,HALF)]
GEOMETRY={
 'base': (BASE_POLY,BASE_HOLES,BASE_T),
 'left_wall':(SIDE_POLY,[],T), 'right_wall':(SIDE_POLY,[],T),
 'front':(GABLE_POLY,[HOLE],T), 'back':(GABLE_POLY,[],T),
 'left_roof':(ROOF_POLY,ROOF_HOLES,T),'right_roof':(ROOF_POLY,ROOF_HOLES,T),
}
R_SIDE=rot_x(math.pi/2)
R_END=np.array([[0.,0,1],[1,0,0],[0,1,0]])
DESIRED={
 'base':(np.array([CX,0,TOP+BASE_T/2]),np.eye(3)),
 'left_wall':(np.array([CX,HALF,TOP+BASE_T]),R_SIDE),
 'right_wall':(np.array([CX,-HALF,TOP+BASE_T]),R_SIDE),
 'front':(np.array([CX-HALF,0,TOP+BASE_T]),R_END),
 'back':(np.array([CX+HALF,0,TOP+BASE_T]),R_END),
}
for name,sign in [('left_roof',1),('right_roof',-1)]:
    R=rot_x(-sign*math.pi/4)
    DESIRED[name]=(np.array([CX,sign*.04,TOP+BASE_T+EAVE+RISE-.04])+R[:,2]*T/2,R)
START=dict(DESIRED)
# Upright loose parts in low slots, not attached to their racks. The racks only
# present the thin panels for grasping; the assembled house rests on its base.
START.update({
 'left_wall':(np.array([.14,.32,TOP+.014]),R_SIDE),
 'right_wall':(np.array([.14,-.32,TOP+.014]),R_SIDE),
 'front':(np.array([.36,.37,TOP+.014]),R_END),
 'back':(np.array([.39,-.30,TOP+.075]),R_SIDE),
 'left_roof':(np.array([.49,.53,TOP+.170]),DESIRED['left_roof'][1]),
 'right_roof':(np.array([.49,-.53,TOP+.170]),DESIRED['right_roof'][1]),
})
PANELS={}
for name,(poly,holes,th) in GEOMETRY.items():
    mesh=panel_mesh(poly,holes,th)
    p,R=START[name]
    PANELS[name]=mk.mesh_part(name,tuple(float(v) for v in p),[mk.Shape(mesh, ROOF if 'roof' in name else WOOD)],
                             mass=float(mesh.volume()*600),quat_xyzw=quat(R),friction=.65,contact_offset=.0005)
OBJECTS=list(PANELS.values())
# Low bench locators restrain the loose base laterally; they do not attach it
# or support any wall/roof. The base remains removable vertically.
for axis in (0,1):
    for sign in (-1,1):
        for cross in (-.060,.060):
            xy=np.array([CX,0.]);xy[axis]+=sign*.100;xy[1-axis]+=cross
            size=(.008,.018,.010) if axis==0 else (.018,.008,.010)
            OBJECTS.append(Box(name=f'base_stop_{axis}_{sign}_{cross}'.replace('-','m').replace('.','p'),
                               pos=(float(xy[0]),float(xy[1]),TOP+.005),size=size,color=(.32,.35,.38)))
# Physical presentation supports: 12 mm deep channels straddle the 6 mm panels.
for name in ['left_wall','right_wall','front','back']:
    p,R=START[name]
    for u in (-.06,.06):
        for n in (-1,1):
            c=p+R@np.array([u,.006,n*.0085])
            h=float(p[2]-TOP+.012)
            c[2]=TOP+h/2
            dims=(.022,.008,h) if name.endswith('wall') or name=='back' else (.008,.022,h)
            OBJECTS.append(Box(name=f'rack_{name}_{u}_{n}'.replace('-','m').replace('.','p'),
                               pos=tuple(float(v) for v in c),size=dims,color=(.32,.35,.38)))
# Raised back-panel rack has a physical shelf below the two tabs.
OBJECTS.append(Box(name='back_rack_shelf',pos=(.39,-.30,TOP+.0305),size=(.150,.026,.061),color=(.32,.35,.38)))
# Roof racks support the tilted panels from below, clear of both the planned
# grip at the ridge and the slots. Two supports prevent yaw without attachment.
for name in ['left_roof','right_roof']:
    p,R=START[name]
    for u in (-.065,.065):
        for v in (-.043,.043):
            c=p+R@np.array([u,v,-T/2])
            h=float(c[2]-TOP-.0055)
            OBJECTS.append(Box(name=f'rack_{name}_{u}_{v}'.replace('-','m').replace('.','p'),
                               pos=(float(c[0]),float(c[1]),TOP+h/2),size=(.014,.010,h),color=(.32,.35,.38)))

# A low eave lip prevents either loose roof sliding off its presentation rack.
for name,sign in [('left_roof',1),('right_roof',-1)]:
    p,R=START[name]; c=p+R@np.array([0,sign*.066,0]);h=float(c[2]-TOP+.008)
    OBJECTS.append(Box(name=f'{name}_rack_lip',pos=(float(c[0]),float(c[1]),TOP+h/2),size=(.140,.006,h),color=(.32,.35,.38)))

@mk.on_reset
def reset_state(environment=None):
    _ST.clear();_ST.update(step=0,last_step=-1,last=(0.,False),history=deque(maxlen=STILL_STEPS),
                          released={n:True for n in PANELS},violation='')
reset_state()

@mk.on_step
def update_step(environment):
    _ST['step']+=1
    pts=finger_points(environment)
    for n,spec in PANELS.items():
        st=mk._mesh_state(environment,spec)
        lo,hi=spec.shapes[0].mesh.bounds()
        _ST['released'][n]=not any(np.all(np.asarray(st.to_local(pt))>=lo-RELEASE)
                                       and np.all(np.asarray(st.to_local(pt))<=hi+RELEASE) for pt in pts)
        if float(st.pos[2]) < TOP-.025:_ST['violation']='panel fell off table'

def relative_error(objects,n):
    b=objects['base'];Rb=b.rotation();bp=np.asarray(b.pos)
    p,R=DESIRED[n];p0,R0=DESIRED['base']
    actual=Rb.T@(np.asarray(objects[n].pos)-bp)
    target=p-p0
    d=float(np.linalg.norm(actual-target))
    rel=Rb.T@objects[n].rotation()
    a=math.acos(float(np.clip((np.trace(R.T@rel)-1)/2,-1,1)))
    return d,a

def seated(objects,n):
    d,a=relative_error(objects,n)
    return d<=.003 and a<=math.radians(2.0) and _ST['released'][n]

def witnesses(objects):
    # Track four body-fixed points so rotation cannot masquerade as stillness.
    return np.array([objects[n].to_world(p) for n in PANELS
                     for p in [(0,0,0),(.05,0,0),(0,.05,0),(0,0,.005)]])

def evaluate(objects):
    base=objects['base']
    if np.linalg.norm(np.asarray(base.pos)[:2]-np.array([CX,0]))>.020:
        _ST['violation']='base moved beyond 20 mm'
    if math.acos(float(np.clip(base.rotation()[2,2],-1,1)))>math.radians(15):
        _ST['violation']='base tipped beyond 15 degrees'
    for n in PANELS:
        if float(objects[n].pos[2])<TOP-.025:_ST['violation']='panel fell off table'
    present={n:seated(objects,n) for n in PANELS if n!='base'}
    base_on_table=(abs(float(base.pos[2])-(TOP+BASE_T/2))<=.001
                   and math.acos(float(np.clip(base.rotation()[2,2],-1,1)))<=math.radians(2.0))
    full=all(present.values()) and all(_ST['released'].values()) and base_on_table
    _ST['history'].append((witnesses(objects),full))
    if _ST['violation']:return 0.,False
    # Full credit and success use the identical 1-second pose-stability gate.
    hist=_ST['history']
    still=(len(hist)==STILL_STEPS and all(f for _,f in hist)
           and max(float(np.linalg.norm(h-hist[-1][0],axis=1).max()) for h,_ in hist)<=.001)
    if full:return (1.,True) if still else (.75,False)
    if all(present[n] for n in ['left_wall','right_wall','front','back']):return .5,False
    if present['left_wall'] and present['right_wall']:return .25,False
    return 0.,False

def once(objects):
    if _ST['last_step']!=_ST['step']:
        _ST['last_step']=_ST['step'];_ST['last']=evaluate(objects)
    return _ST['last']

def success(objects):return once(objects)[1]
def score(objects):return once(objects)[0]
def distance(objects):return sum(relative_error(objects,n)[0] for n in PANELS if n!='base')
def demo():return []
