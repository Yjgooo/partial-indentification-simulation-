#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, os, time, json
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds

RHO=0.10
SAMPLE_SIZES=(4_000,16_000,64_000,256_000,1_024_000)
TINY=1e-12

@dataclass
class DGP:
    K:int; px:np.ndarray; propensity:np.ndarray; outcome:np.ndarray
    alpha:np.ndarray; p:np.ndarray; g:np.ndarray; q:np.ndarray
    main:np.ndarray; aux:np.ndarray; true_ate:float

def make_dgp(K:int)->DGP:
    z=np.linspace(-1.0,1.0,K)
    z=z/np.sqrt(np.mean(z**2))
    px=np.ones(K)/K
    propensity=0.5+0.1*z
    y0=0.35+0.15*z
    y1=y0+0.08
    outcome=np.vstack([y0,y1])
    alpha1=float(px@propensity); alpha=np.array([1-alpha1,alpha1])
    g=np.empty((2,K)); p=np.empty(2); q=np.empty((2,2,K))
    for a in range(2):
        pa_x=propensity if a==1 else 1-propensity
        g[a]=px*pa_x/alpha[a]
        p[a]=float(g[a]@outcome[a])
        q[a,1]=g[a]*outcome[a]/p[a]
        q[a,0]=g[a]*(1-outcome[a])/(1-p[a])
    main=np.empty((2,2,K))
    for a in range(2):
        for y in range(2):
            py=p[a] if y==1 else 1-p[a]
            main[a,y]=alpha[a]*py*q[a,y]
    aux=alpha[:,None]*g
    true=float(px@(y1-y0))
    assert abs(main.sum()-1)<1e-12 and abs(aux.sum()-1)<1e-12
    return DGP(K,px,propensity,outcome,alpha,p,g,q,main,aux,true)

def smoothed_probability(counts:np.ndarray,smooth:float=0.5)->np.ndarray:
    counts=np.asarray(counts,float)
    return (counts+smooth)/(counts.sum()+smooth*counts.size)

def estimate_main(counts:np.ndarray):
    law=smoothed_probability(counts)
    alpha=law.sum(axis=(1,2))
    p=law[:,1].sum(axis=1)/alpha
    q=law/law.sum(axis=2,keepdims=True)
    return alpha,p,q

def estimate_aux(counts:np.ndarray):
    law=smoothed_probability(counts)
    return law/law.sum(axis=1,keepdims=True)

def direct_aux_pair(alpha,p,g,tilde_q,rho=RHO):
    K=g.shape[1]; nq=4*K; nu=4*K; n=nq+nu
    def qi(a,y,k): return (2*a+y)*K+k
    def ui(a,y,k): return nq+(2*a+y)*K+k
    px=alpha[0]*g[0]+alpha[1]*g[1]
    beta=np.zeros(n)
    for k in range(K):
        beta[qi(1,1,k)]=px[k]*p[1]/max(g[1,k],TINY)
        beta[qi(0,1,k)]=-px[k]*p[0]/max(g[0,k],TINY)
    Aeq=[];beq=[]
    for a in range(2):
        for y in range(2):
            row=np.zeros(n);row[[qi(a,y,k) for k in range(K)]]=1
            Aeq.append(row);beq.append(1.)
        for k in range(K):
            row=np.zeros(n);row[qi(a,0,k)]=1-p[a];row[qi(a,1,k)]=p[a]
            Aeq.append(row);beq.append(g[a,k])
    Aub=[];bub=[]
    for a in range(2):
        for y in range(2):
            for k in range(K):
                row=np.zeros(n);row[qi(a,y,k)]=1;row[ui(a,y,k)]=-1
                Aub.append(row);bub.append(tilde_q[a,y,k])
                row=np.zeros(n);row[qi(a,y,k)]=-1;row[ui(a,y,k)]=-1
                Aub.append(row);bub.append(-tilde_q[a,y,k])
            row=np.zeros(n);row[[ui(a,y,k) for k in range(K)]]=1
            Aub.append(row);bub.append(2*rho)
    bounds=[(0.,1.)]*nq+[(0.,None)]*nu
    kw=dict(A_ub=np.asarray(Aub),b_ub=np.asarray(bub),A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    lo=linprog(beta,**kw); up=linprog(-beta,**kw)
    if not (lo.success and up.success): return np.nan,np.nan,False
    return float(lo.fun),float(-up.fun),True

def endpoint_system(alpha,p,g,tilde_q,rho,sign):
    K=g.shape[1];nq=4*K;nu=4*K;nx=1+nq+nu
    def qi(a,y,k): return 1+(2*a+y)*K+k
    def ui(a,y,k): return 1+nq+(2*a+y)*K+k
    px=alpha[0]*g[0]+alpha[1]*g[1]
    beta=np.zeros(nx)
    for k in range(K):
        beta[qi(1,1,k)]=px[k]*p[1]/max(g[1,k],TINY)
        beta[qi(0,1,k)]=-px[k]*p[0]/max(g[0,k],TINY)
    M=[];rhs=[]
    row=np.zeros(nx);row[0]=1;row-=sign*beta;M.append(row);rhs.append(0.)
    for a in range(2):
        for y in range(2):
            row=np.zeros(nx);row[[qi(a,y,k) for k in range(K)]]=1
            M.extend([row,-row]);rhs.extend([1.,-1.])
        for k in range(K):
            row=np.zeros(nx);row[qi(a,0,k)]=1-p[a];row[qi(a,1,k)]=p[a]
            M.extend([row,-row]);rhs.extend([g[a,k],-g[a,k]])
    for a in range(2):
        for y in range(2):
            for k in range(K):
                row=np.zeros(nx);row[ui(a,y,k)]=1;row[qi(a,y,k)]=-1;M.append(row);rhs.append(-tilde_q[a,y,k])
                row=np.zeros(nx);row[ui(a,y,k)]=1;row[qi(a,y,k)]=1;M.append(row);rhs.append(tilde_q[a,y,k])
            row=np.zeros(nx);row[[ui(a,y,k) for k in range(K)]]=-1;M.append(row);rhs.append(-2*rho)
    lower=np.r_[-1.,np.zeros(nq+nu)];upper=np.r_[1.,np.ones(nq+nu)]
    for j in range(nx):
        row=np.zeros(nx);row[j]=1;M.append(row);rhs.append(lower[j]);M.append(-row);rhs.append(-upper[j])
    bounds=[(-1.25,1.25)]+[(-0.25,1.25)]*(nq+nu)
    return np.asarray(M),np.asarray(rhs),bounds

def penalty_level(n_main,n_aux,multiplier=1.):
    neff=min(n_main,n_aux)
    return multiplier*5*math.log(math.log(float(neff)))/math.log(math.log(30000.))

def solve_debiased(M,rhs,bounds_x,w_n,face_tol):
    norms=np.linalg.norm(M,axis=1)
    if np.any(norms<=TINY): return None
    Mn=M/norms[:,None];cn=rhs/norms;Mn[0]=M[0];cn[0]=rhs[0]
    qcon,nx=M.shape;weights=np.full(qcon,w_n);weights[0]=1.
    pobj=np.r_[1.,np.zeros(nx-1),weights]
    Aub=np.hstack([-Mn,-np.eye(qcon)]);bub=-cn
    bounds=bounds_x+[(0.,None)]*qcon
    opts={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    s1=linprog(pobj,A_ub=Aub,b_ub=bub,bounds=bounds,method='highs',options=opts)
    if not s1.success:return None
    s2obj=np.zeros(nx+qcon);s2obj[0]=-1
    s2=linprog(s2obj,A_ub=Aub,b_ub=bub,A_eq=pobj[None,:],b_eq=np.array([s1.fun]),bounds=bounds,method='highs',options=opts)
    if not s2.success:
        faceA=np.vstack([Aub,pobj,-pobj]);faceb=np.r_[bub,s1.fun+face_tol,-s1.fun+face_tol]
        s2=linprog(s2obj,A_ub=faceA,b_ub=faceb,bounds=bounds,method='highs',options=opts)
    if not s2.success:return None
    x=s2.x[:nx];sl=s2.x[nx:]
    return {'selected':float(s2.x[0]),'slack':float(sl.sum()),'gap':float(pobj@s2.x-s1.fun),'violation':float(np.maximum(rhs-M@x,0).max(initial=0.))}

def penalty_pair(alpha,p,g,q,rho,n):
    w=penalty_level(n,n);ft=min(1e-8,n**(-0.75))
    lo=solve_debiased(*endpoint_system(alpha,p,g,q,rho,1.),w,ft)
    nu=solve_debiased(*endpoint_system(alpha,p,g,q,rho,-1.),w,ft)
    return lo,nu,w

def noaux_value_grad(qflat,dgp:DGP):
    K=dgp.K;q=qflat.reshape(2,2,K);p=dgp.p;a=dgp.alpha
    g=p[:,None]*q[:,1]+(1-p)[:,None]*q[:,0];r=p[:,None]*q[:,1];m=r/g
    px=a[0]*g[0]+a[1]*g[1];diff=m[1]-m[0]
    val=float(np.sum(px*diff));gr=np.zeros_like(q)
    gr[1,1]=a[1]*p[1]*diff+px*(p[1]*(1-m[1])/g[1])
    gr[1,0]=a[1]*(1-p[1])*diff-px*((1-p[1])*m[1]/g[1])
    gr[0,1]=a[0]*p[0]*diff-px*(p[0]*(1-m[0])/g[0])
    gr[0,0]=a[0]*(1-p[0])*diff+px*((1-p[0])*m[0]/g[0])
    return val,gr.reshape(-1)

def noaux_constraints(dgp:DGP,rho=RHO,eps=1e-9):
    K=dgp.K;nq=4*K;n=2*nq;c=dgp.q.reshape(-1)
    Aeq=np.zeros((4,n))
    for s in range(4):Aeq[s,s*K:(s+1)*K]=1
    Aub=[];bub=[]
    for i in range(nq):
        row=np.zeros(n);row[i]=1;row[nq+i]=-1;Aub.append(row);bub.append(c[i])
        row=np.zeros(n);row[i]=-1;row[nq+i]=-1;Aub.append(row);bub.append(-c[i])
    for s in range(4):
        row=np.zeros(n);row[nq+s*K:nq+(s+1)*K]=1;Aub.append(row);bub.append(2*rho)
    return (LinearConstraint(Aeq,np.ones(4),np.ones(4)),LinearConstraint(np.asarray(Aub),-np.inf,np.asarray(bub)),Bounds(np.r_[np.full(nq,eps),np.zeros(nq)],np.r_[np.ones(nq),np.full(nq,np.inf)]))

def noaux_random_start(dgp,rng,rho=RHO):
    K=dgp.K;qs=[]
    for c in dgp.q.reshape(4,K):
        r=rng.dirichlet(np.ones(K));lam=rng.uniform(0,rho);qs.append((1-lam)*c+lam*r)
    q=np.asarray(qs).reshape(-1);return np.r_[q,np.abs(q-dgp.q.reshape(-1))]

def approximate_noaux(dgp:DGP,starts=32,seed=2026090719):
    K=dgp.K;nq=4*K;cons=noaux_constraints(dgp);rng=np.random.default_rng(seed)
    rows={}
    for endpoint,sign in [('lower',1.),('upper',-1.)]:
        best=None;values=[]
        def fun(x):return sign*noaux_value_grad(x[:nq],dgp)[0]
        def jac(x):return np.r_[sign*noaux_value_grad(x[:nq],dgp)[1],np.zeros(nq)]
        for j in range(starts):
            x0=np.r_[dgp.q.reshape(-1),np.zeros(nq)] if j==0 else noaux_random_start(dgp,rng)
            res=minimize(fun,x0,jac=jac,method='SLSQP',bounds=cons[2],constraints=cons[:2],options={'ftol':1e-11,'maxiter':1500,'disp':False})
            if res.success:
                val=sign*float(res.fun);values.append(val)
                if best is None or (val<best if endpoint=='lower' else val>best):best=val
        if best is None:raise RuntimeError('noaux optimization failed')
        rows[endpoint]={'value':best,'successful_starts':len(values),'best_5':sorted(values)[:5] if endpoint=='lower' else sorted(values,reverse=True)[:5]}
    return rows

def population_oracle(K,noaux_starts=32):
    d=make_dgp(K);lo,up,ok=direct_aux_pair(d.alpha,d.p,d.g,d.q,RHO)
    if not ok:raise RuntimeError('population aux LP failed')
    noaux=approximate_noaux(d,noaux_starts) if noaux_starts>0 else {'lower':{'value':np.nan},'upper':{'value':np.nan}}
    return {'K':K,'rho':RHO,'true_ate':d.true_ate,'aux_lower':lo,'aux_upper':up,'aux_width':up-lo,'noaux_lower_best_found':noaux['lower']['value'],'noaux_upper_best_found':noaux['upper']['value'],'noaux_width_best_found':noaux['upper']['value']-noaux['lower']['value'],'noaux_diagnostics':noaux}

def pop_exactness(K,n,oracle):
    d=make_dgp(K);lo,nu,w=penalty_pair(d.alpha,d.p,d.g,d.q,RHO,n)
    if lo is None or nu is None:return {'n':n,'success':False}
    l=lo['selected'];u=-nu['selected'];err=max(abs(l-oracle['aux_lower']),abs(u-oracle['aux_upper']))
    return {'n':n,'penalty_level':w,'lower':l,'upper':u,'max_abs_error':err,'success':err<=1e-7}

def simulate_one(payload):
    K,n,rep,seed,oracle=payload;start=time.perf_counter();d=make_dgp(K)
    ss=np.random.SeedSequence(seed);ms,as_=ss.spawn(2);mr=np.random.default_rng(ms);ar=np.random.default_rng(as_)
    mc=mr.multinomial(n,d.main.ravel()).reshape(2,2,K);ac=ar.multinomial(n,d.aux.ravel()).reshape(2,K)
    alpha,p,q=estimate_main(mc);g=estimate_aux(ac)
    lo,nu,w=penalty_pair(alpha,p,g,q,RHO,n)
    rawlo,rawup,rawok=direct_aux_pair(alpha,p,g,q,RHO)
    if lo is None or nu is None:return {'K':K,'n':n,'rep':rep,'seed':seed,'success':False,'runtime_seconds':time.perf_counter()-start}
    lh=lo['selected'];uh=-nu['selected'];le=lh-oracle['aux_lower'];ue=uh-oracle['aux_upper'];joint=max(abs(le),abs(ue))
    return {'K':K,'n':n,'rep':rep,'seed':seed,'success':True,'coherent':lh<=uh+1e-10,'raw_feasible':rawok,'lower_hat':lh,'upper_hat':uh,'width_hat':uh-lh,'lower_error':le,'upper_error':ue,'joint_abs_error':joint,'scaled_lower_error':math.sqrt(n)*le,'scaled_upper_error':math.sqrt(n)*ue,'scaled_joint_abs_error':math.sqrt(n)*joint,'lower_total_slack':lo['slack'],'upper_total_slack':nu['slack'],'lower_primary_gap':lo['gap'],'upper_primary_gap':nu['gap'],'runtime_seconds':time.perf_counter()-start}

def summarize(df):
    rows=[]
    for n,c in df[df.success].groupby('n'):
        rm=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
        rows.append({'n':int(n),'repetitions':len(c),'success_rate':float(df[df.n==n].success.mean()),'coherent_rate':float(c.coherent.mean()),'raw_feasibility_rate':float(c.raw_feasible.mean()),'mean_lower':float(c.lower_hat.mean()),'mean_upper':float(c.upper_hat.mean()),'mean_width':float(c.width_hat.mean()),'lower_bias':float(c.lower_error.mean()),'upper_bias':float(c.upper_error.mean()),'lower_rmse':rm(c.lower_error),'upper_rmse':rm(c.upper_error),'joint_q95':float(c.joint_abs_error.quantile(.95)),'scaled_lower_rmse':math.sqrt(n)*rm(c.lower_error),'scaled_upper_rmse':math.sqrt(n)*rm(c.upper_error),'scaled_joint_q95':float(c.scaled_joint_abs_error.quantile(.95)),'median_runtime_seconds':float(c.runtime_seconds.median()),'mean_lower_slack':float(c.lower_total_slack.mean()),'mean_upper_slack':float(c.upper_total_slack.mean())})
    out=pd.DataFrame(rows)
    for metric in ['lower_rmse','upper_rmse','joint_q95']:
        out.attrs[metric+'_decay_exponent']=float(-np.polyfit(np.log(out.n),np.log(out[metric]),1)[0])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--K',type=int,required=True);ap.add_argument('--repetitions',type=int,default=1500);ap.add_argument('--workers',type=int,default=min(6,os.cpu_count() or 1));ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base-seed',type=int,default=2026090711);ap.add_argument('--noaux-starts',type=int,default=32);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    oracle=population_oracle(args.K,args.noaux_starts);(args.output/'oracle.json').write_text(json.dumps(oracle,indent=2)+'\n')
    exact=pd.DataFrame([pop_exactness(args.K,n,oracle) for n in SAMPLE_SIZES]);exact.to_csv(args.output/'population_penalty_exactness.csv',index=False)
    if not exact.success.all():raise RuntimeError('population exactness failed')
    allrows=[];t0=time.time()
    for ci,n in enumerate(SAMPLE_SIZES):
        payloads=[(args.K,n,r,args.base_seed+args.K*100_000_000+ci*10_000_000+r*7919,oracle) for r in range(args.repetitions)]
        ct=time.time()
        if args.workers==1:rows=[simulate_one(x) for x in payloads]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex: rows=list(ex.map(simulate_one,payloads,chunksize=4))
        fr=pd.DataFrame(rows);fr.to_csv(args.output/f'replications_n{n}.csv.gz',index=False,compression='gzip');allrows+=rows
        print(f'K={args.K} n={n}: {int(fr.success.sum())}/{len(fr)} success, raw feasible={fr.raw_feasible.mean():.3f}, elapsed={time.time()-ct:.1f}s',flush=True)
    df=pd.DataFrame(allrows);df.to_csv(args.output/'replications.csv.gz',index=False,compression='gzip');summ=summarize(df);summ.to_csv(args.output/'summary.csv',index=False)
    meta={'K':args.K,'repetitions':args.repetitions,'sample_sizes':SAMPLE_SIZES,'rho':RHO,'elapsed_seconds':time.time()-t0,'decay_exponents':{k:v for k,v in summ.attrs.items()}}
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(summ.to_string(index=False),flush=True);print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
