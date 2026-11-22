"""Export full-scale plywood profiles; SVG units are millimetres. Run locally."""
from pathlib import Path
import json
import task
out=Path(__file__).parent/'fabrication'
out.mkdir(exist_ok=True)
for name,(poly,holes,th) in task.GEOMETRY.items():
    rings=[poly]+list(holes)
    xs=[p[0]*1000 for r in rings for p in r];ys=[p[1]*1000 for r in rings for p in r]
    x0,y0=min(xs)-10,min(ys)-10;w,h=max(xs)-x0+10,max(ys)-y0+10
    def path(r):return 'M '+' L '.join(f'{x*1000:.4f},{-y*1000:.4f}' for x,y in r)+' Z'
    # Flipping y in the view box keeps the same right-handed physical profile.
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.4f}mm" height="{h:.4f}mm" viewBox="{x0:.4f} {-max(ys)-10:.4f} {w:.4f} {h:.4f}">'
    svg+='<path d="'+' '.join(path(r) for r in rings)+'" fill="none" stroke="black" stroke-width="0.15"/>'
    svg+='</svg>'
    (out/(name+'.svg')).write_text(svg)
(out/'dimensions.json').write_text(json.dumps({n:{'thickness_mm':th*1000,'bottom_tab_bevel':('linear 16 x 6 mm shoulder to 12 x 2 mm tip over 12 mm' if len(p)>4 else None),'outline_mm':[[u*1000,v*1000] for u,v in p],
    'holes_mm':[[[u*1000,v*1000] for u,v in ring] for ring in holes]} for n,(p,holes,th) in task.GEOMETRY.items()},indent=2))

(out/'rack_layout.json').write_text(json.dumps({
    'units':'millimetres; z above tabletop',
    'supports':[{'name':o.name,'center_mm':[o.pos[0]*1000,o.pos[1]*1000,(o.pos[2]-task.TOP)*1000],
                 'size_mm':[v*1000 for v in o.size]} for o in task.OBJECTS if o.name not in task.PANELS],
    'initial_panels':{n:{'origin_mm':[p[0]*1000,p[1]*1000,(p[2]-task.TOP)*1000],'rotation_matrix':R.tolist()}
                      for n,(p,R) in task.START.items()},
    'assembled_panels':{n:{'origin_mm':[p[0]*1000,p[1]*1000,(p[2]-task.TOP)*1000],'rotation_matrix':R.tolist()}
                        for n,(p,R) in task.DESIRED.items()}
},indent=2))
