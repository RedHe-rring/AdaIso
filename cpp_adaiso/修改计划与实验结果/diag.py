import argparse
from pathlib import Path
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

outdir=Path('cpp_adaiso/results/diagnostics')
outdir.mkdir(parents=True, exist_ok=True)
kCsfEta=1e-4; kMinEdgeLength=1e-12

def read_off(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        assert f.readline().strip()=='OFF'
        line=f.readline().strip()
        while not line or line.startswith('#'): line=f.readline().strip()
        nv,nf,*_=map(int,line.split())
        V=np.array([tuple(map(float,f.readline().split()[:3])) for _ in range(nv)],dtype=float)
        F=[]
        for _ in range(nf):
            vals=list(map(int,f.readline().split()))
            F.append(tuple(vals[1:4]))
        return V,F

def read_ply(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = None
        nv = None
        nf = 0
        current_element = None
        vertex_props = []

        # read header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")

            line = line.strip()
            if not line or line.startswith("comment"):
                continue

            parts = line.split()

            if parts[0] == "format":
                fmt = parts[1]
                if fmt != "ascii":
                    raise ValueError(
                        f"{path} is {fmt} PLY. This read_ply only supports ascii PLY."
                    )

            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    nv = int(parts[2])
                elif current_element == "face":
                    nf = int(parts[2])

            elif parts[0] == "property":
                if current_element == "vertex":
                    # Example:
                    # property float x
                    # property uchar red
                    if len(parts) >= 3 and parts[1] != "list":
                        vertex_props.append(parts[-1])

            elif parts[0] == "end_header":
                break

        if nv is None:
            raise ValueError(f"{path} has no vertex element")

        try:
            ix = vertex_props.index("x")
            iy = vertex_props.index("y")
            iz = vertex_props.index("z")
        except ValueError as e:
            raise ValueError(
                f"{path} vertex properties do not contain x/y/z. "
                f"Got properties: {vertex_props}"
            ) from e

        # read vertices
        V = np.empty((nv, 3), dtype=float)

        for i in range(nv):
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended while reading vertices")

            vals = line.split()
            V[i, 0] = float(vals[ix])
            V[i, 1] = float(vals[iy])
            V[i, 2] = float(vals[iz])

        # read faces
        F = []

        for _ in range(nf):
            line = f.readline()
            if not line:
                break

            vals = line.split()
            if not vals:
                continue

            n = int(vals[0])
            ids = list(map(int, vals[1:1 + n]))

            if n == 3:
                F.append(tuple(ids))
            elif n > 3:
                # fan triangulation: [0,1,2], [0,2,3], ...
                for j in range(1, n - 1):
                    F.append((ids[0], ids[j], ids[j + 1]))

        return V, F

def read_obj(path):
    V=[]; F=[]
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            if line.startswith('v '): V.append(tuple(map(float,line.split()[1:4])))
            elif line.startswith('f '): F.append(tuple(int(tok.split('/')[0])-1 for tok in line.split()[1:4]))
    return np.array(V,dtype=float),F

def read_mesh(path):
    path=Path(path)
    suffix=path.suffix.lower()
    if suffix=='.off': return read_off(path)
    if suffix=='.ply': return read_ply(path)
    if suffix=='.obj': return read_obj(path)
    raise ValueError(f'unsupported mesh extension: {path}')

def write_ply(path,V,F,colors):
    colors=np.clip(np.asarray(colors),0,255).astype(int)
    with open(path,'w',encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(V)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n')
        for p,c in zip(V,colors):
            f.write(f'{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
        for a,b,c in F:
            f.write(f'3 {a} {b} {c}\n')

def cmap(values, invert=False):
    v=np.asarray(values,dtype=float)
    lo,hi=np.percentile(v[np.isfinite(v)], [2,98])
    if hi<=lo: hi=lo+1.0
    t=np.clip((v-lo)/(hi-lo),0,1)
    if invert: t=1-t
    # blue -> cyan -> yellow -> red
    stops=np.array([[49,54,149],[69,117,180],[116,173,209],[171,217,233],[224,243,248],[254,224,144],[253,174,97],[244,109,67],[215,48,39],[165,0,38]],float)
    x=t*(len(stops)-1); i=np.floor(x).astype(int); j=np.clip(i+1,0,len(stops)-1); a=(x-i)[:,None]
    return stops[i]*(1-a)+stops[j]*a

def norm(v):
    n=float(np.linalg.norm(v)); return v/n if n>1e-14 and math.isfinite(n) else np.zeros(3)

def edge_key(a,b): return (a,b) if a<b else (b,a)

def edges_neigh(F,nv):
    E=set(); neigh=[set() for _ in range(nv)]
    for a,b,c in F:
        for u,v in ((a,b),(b,c),(c,a)):
            E.add(edge_key(u,v)); neigh[u].add(v); neigh[v].add(u)
    return E,neigh

def vertex_normals(V,F):
    N=np.zeros_like(V)
    for a,b,c in F:
        cr=np.cross(V[b]-V[a],V[c]-V[a]); area=0.5*np.linalg.norm(cr); fn=norm(cr); w=fn*area
        N[a]+=w; N[b]+=w; N[c]+=w
    return np.array([norm(n) for n in N])

def cot_weights(V,F):
    W={}
    def cot_at(a,b,c):
        u=V[b]-V[a]; v=V[c]-V[a]; d=float(np.linalg.norm(np.cross(u,v)))
        return 0.0 if d<=1e-14 else float(np.dot(u,v))/d
    for a,b,c in F:
        for e,val in ((edge_key(b,c),cot_at(a,b,c)),(edge_key(c,a),cot_at(b,c,a)),(edge_key(a,b),cot_at(c,a,b))):
            W[e]=W.get(e,0.0)+val
    for (a,b),w in list(W.items()):
        w*=0.5
        if not math.isfinite(w) or w<=0: w=kCsfEta
        W[(a,b)]=w
    return W

def vertex_masses(V,F):
    m=np.zeros(len(V),dtype=float)
    for a,b,c in F:
        area=0.5*np.linalg.norm(np.cross(V[b]-V[a],V[c]-V[a]))
        share=area/3.0
        m[a]+=share; m[b]+=share; m[c]+=share
    if len(m)==0 or np.any(~np.isfinite(m)) or np.any(m<=0):
        raise ValueError('screened CSF requires positive finite vertex masses')
    return m

def solve_screened(V,F,init,alpha=0.5):
    if len(V)==0:
        raise ValueError('screened CSF requires a non-empty mesh')
    if alpha<=0 or not math.isfinite(alpha):
        raise ValueError('screened CSF alpha must be positive')
    adj=[[] for _ in range(len(V))]
    for (a,b),w in cot_weights(V,F).items(): adj[a].append((b,w)); adj[b].append((a,w))
    for ns in adj: ns.sort()
    E,_=edges_neigh(F,len(V))
    if not E:
        raise ValueError('screened CSF requires at least one edge')
    h=sum(np.linalg.norm(V[a]-V[b]) for a,b in E)/len(E)
    if h<=0 or not math.isfinite(h):
        raise ValueError('screened CSF requires positive mean edge length')
    lam=alpha*h*h
    masses=vertex_masses(V,F)
    rows=[]; cols=[]; data=[]; rhs=masses*init
    for i in range(len(V)):
        diag=masses[i]
        for n,w in adj[i]:
            diag+=lam*w
            rows.append(i); cols.append(n); data.append(-lam*w)
        rows.append(i); cols.append(i); data.append(diag)
    A=coo_matrix((data,(rows,cols)),shape=(len(V),len(V))).tocsr()
    sol=spsolve(A,rhs)
    if np.any(~np.isfinite(sol)):
        raise ValueError('screened CSF solve produced non-finite values')
    return np.asarray(sol,dtype=float)

def histogram(curv):
    multipliers=np.array([1.8,1.4,1.0,0.8,0.6])
    mn=curv.min(); mx=curv.max(); bins=40; width=(mx-mn)/bins; counts=np.zeros(bins,int)
    for v in curv:
        b=max(0,min(bins-1,int(math.floor((v-mn)/width)))); counts[b]+=1
    maxbins=[i for i,c in enumerate(counts) if c==counts.max()]; base=mn+(maxbins[len(maxbins)//2]+0.5)*width
    s=np.sort(curv); idx=int(np.searchsorted(s,base,'left')); idx=max(1,min(idx,len(s)-1)); unit=max(1,(len(s)-idx)//5)
    ci=lambda x:max(0,min(int(x),len(s)-1)); cuts=[s[ci((idx//5)*2)],s[ci((idx//5)*4)],s[ci(idx+unit)],s[ci(idx+unit*3)]]
    h=np.ones(len(curv))
    h[curv<cuts[0]]=1.8; h[(curv>=cuts[0])&(curv<cuts[1])]=1.4; h[(curv>=cuts[1])&(curv<cuts[2])]=1.0; h[(curv>=cuts[2])&(curv<cuts[3])]=0.8; h[curv>=cuts[3]]=0.6
    return h

def gaussian_values(points,Vin,lengths):
    tree=KDTree(Vin); d,idx=tree.query(points,k=min(8,len(Vin)))
    out=[]
    for di,ii in zip(np.atleast_2d(d),np.atleast_2d(idx)):
        dk=di[-1]; dkprev=di[-2] if len(di)>1 else dk; h=max(dk+max(0.0,dk-dkprev),1e-12)
        w=np.exp(-((di/h)**2)); out.append(float((w*lengths[ii]).sum()/w.sum()))
    return np.array(out)

def compute_raw_csf_target(Vin,Fin,alpha):
    Ein,neigh=edges_neigh(Fin,len(Vin)); N=vertex_normals(Vin,Fin); raw=np.zeros(len(Vin))
    for i,ns in enumerate(neigh):
        if ns:
            raw[i]=sum(math.acos(max(-1,min(1,float(np.dot(N[i],N[j]))))) for j in ns)/len(ns)
    csf=solve_screened(Vin,Fin,raw,alpha=alpha)
    base=sum(np.linalg.norm(Vin[a]-Vin[b]) for a,b in Ein)/len(Ein)
    target=base*histogram(csf)
    return raw,csf,target

def main():
    parser=argparse.ArgumentParser(description='Write v2.3.0 CSF diagnostic heat PLY files.')
    parser.add_argument('input', help='input mesh: OFF/PLY/OBJ')
    parser.add_argument('--output', help='optional remeshed output mesh for output target heat')
    parser.add_argument('--outdir', default='cpp_adaiso/results/diagnostics')
    parser.add_argument('--name', help='output filename prefix; default is input stem')
    parser.add_argument('--alpha', type=float, default=0.5)
    args=parser.parse_args()

    outdir=Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name=args.name or Path(args.input).stem
    Vin,Fin=read_mesh(args.input)
    raw,csf,target=compute_raw_csf_target(Vin,Fin,args.alpha)

    paths=[
        outdir/f'{name}_input_raw_curv_heat.ply',
        outdir/f'{name}_input_csf_heat.ply',
        outdir/f'{name}_input_target_heat.ply',
    ]
    write_ply(paths[0], Vin, Fin, cmap(raw, invert=False))
    write_ply(paths[1], Vin, Fin, cmap(csf, invert=False))
    write_ply(paths[2], Vin, Fin, cmap(target, invert=True))

    if args.output:
        Vout,Fout=read_mesh(args.output)
        out_target=gaussian_values(Vout,Vin,target)
        out_path=outdir/f'{name}_output_target_heat.ply'
        write_ply(out_path, Vout, Fout, cmap(out_target, invert=True))
        paths.append(out_path)

    print(f'{name}: raw min/max={raw.min():.8g}/{raw.max():.8g}, csf min/max={csf.min():.8g}/{csf.max():.8g}')
    for path in paths:
        print(path)

if __name__ == '__main__':
    main()
