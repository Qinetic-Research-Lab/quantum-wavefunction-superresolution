import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows=list(csv.DictReader(open(r"C:\Users\hello\Quantum\outputs\analysis\test_metrics_canonical.csv")))
om=np.array([float(r['omega']) for r in rows])
ci=1-np.array([float(r['F_cnn']) for r in rows])
si=1-np.array([float(r['F_spline']) for r in rows])
o_=np.argsort(om); om,ci,si=om[o_],ci[o_],si[o_]

# crossover: last spline win and first of the unbroken CNN-win run
wins=ci<si
last_spline_win=om[~wins][-1] if (~wins).any() else None
# find start of final all-win run
idx=len(wins)-1
while idx>0 and wins[idx-1]: idx-=1
print("last spline win at omega =",f"{om[np.where(~wins)[0][-1]]:.3f}")
print("unbroken CNN-win run starts at omega =",f"{om[idx]:.3f}")
print("wins total:",int(wins.sum()))

edges=[0.5,1.4,2.3,3.2,4.1,5.001]; ctr=[];bc=[];bs=[]
for a,b in zip(edges,edges[1:]):
    m=(om>=a)&(om<b)
    if m.sum(): ctr.append(om[m].mean()); bc.append(ci[m].mean()); bs.append(si[m].mean())

plt.rcParams.update({'font.size':6.5,'axes.linewidth':0.8,'font.family':'DejaVu Sans'})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(5.5,2.0),gridspec_kw={'width_ratios':[1.35,1]})
ax1.axvspan(1.95,2.15,color='0.88',zorder=0)
ax1.scatter(om,si,s=7,marker='o',facecolors='none',edgecolors='#B2182B',linewidths=0.5,alpha=.65,zorder=2)
ax1.scatter(om,ci,s=7,marker='^',facecolors='none',edgecolors='#2166AC',linewidths=0.5,alpha=.65,zorder=2)
ax1.plot(ctr,bs,'-o',color='#B2182B',lw=1.3,ms=3.4,zorder=4,label='Cubic spline (bin mean)')
ax1.plot(ctr,bc,'--^',color='#2166AC',lw=1.3,ms=3.4,zorder=4,label='CNN (bin mean)')
ax1.set_yscale('log'); ax1.set_xlabel(r'oscillator frequency  $\omega$')
ax1.set_ylabel(r'infidelity  $1-F$'); ax1.set_xlim(0.3,5.2)
ax1.text(2.05,ax1.get_ylim()[1]*0.55,'crossover',ha='center',fontsize=6,color='0.35')
ax1.legend(frameon=False,fontsize=6,loc='lower right')
ax1.grid(True,which='major',alpha=.25,lw=.5); ax1.set_title('(a) error vs. frequency',fontsize=7.5,loc='left')
ratio=si/ci
ax2.axhline(1.0,color='0.3',lw=1.0,ls=':')
ax2.scatter(om,ratio,s=7,marker='s',facecolors='none',edgecolors='#4D4D4D',linewidths=.5,alpha=.7)
ax2.set_yscale('log'); ax2.set_xlabel(r'oscillator frequency  $\omega$')
ax2.set_ylabel(r'spline infidelity / CNN infidelity'); ax2.set_xlim(0.3,5.2)
ax2.text(4.9,1.25,'CNN lower error',ha='right',fontsize=6,color='0.3')
ax2.text(4.9,0.66,'spline lower error',ha='right',fontsize=6,color='0.3')
ax2.grid(True,which='major',alpha=.25,lw=.5); ax2.set_title('(b) paired ratio',fontsize=7.5,loc='left')
fig.tight_layout()
fig.savefig(r"C:\Users\hello\AppData\Local\Temp\claude\C--Users-hello-OneDrive-Desktop\4b702ca4-dea5-479c-8b03-abb14a7c9f65\scratchpad\figure1_col.png",dpi=300)
fig.savefig(r"C:\Users\hello\AppData\Local\Temp\claude\C--Users-hello-OneDrive-Desktop\4b702ca4-dea5-479c-8b03-abb14a7c9f65\scratchpad\figure1_col.pdf")
print("saved")
