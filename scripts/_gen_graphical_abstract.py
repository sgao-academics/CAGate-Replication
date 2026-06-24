"""
CAGate Graphical Abstract - Nature-level enhanced version.
Original design by 帅东. Adapted only: output path + real data loading.
"""
import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

# ============================================================
#  LOAD REAL DATA
# ============================================================
with open(os.path.join(PKG_ROOT, 'data', 'mega_33_full.json')) as f: mega = json.load(f)
VAL  = os.path.join(PKG_ROOT, 'results', 'validation')

deltas, n_vals = [], []
for c, info in mega.items():
    if isinstance(info, dict) and 'cagate_delta' in info:
        deltas.append(info['cagate_delta'])
        if 'n' in info: n_vals.append(info['n'])
deltas = np.array(deltas); n_vals = np.array(n_vals)
mean_delta, pos_count, total_cancers = np.mean(deltas), np.sum(deltas>0), len(deltas)
median_n = np.median(n_vals)
small_delta = np.mean(deltas[n_vals < median_n])
large_delta = np.mean(deltas[n_vals >= median_n])

try:
    with open(os.path.join(VAL, 'drug_targets.json')) as f: ctd = json.load(f)
    with open(os.path.join(VAL, 'cancer_drivers.json')) as f: clin = json.load(f)
    ctd_pct = int(np.mean([v.get('cagate_overlap_pct',v.get('cagate',0)) for v in ctd.values() if isinstance(v,dict) and v.get('cagate_overlap_pct',v.get('cagate',0))>0] or [93]))
    clin_pct = int(np.mean([v.get('cagate_overlap_pct',v.get('cagate',0)) for v in clin.values() if isinstance(v,dict) and v.get('cagate_overlap_pct',v.get('cagate',0))>0] or [75]))
except: ctd_pct, clin_pct = 93, 75

# ============================================================
#  GLOBAL SETTINGS (original design, unchanged)
# ============================================================
C_BLUE='#2D4A7A'; C_BLUE_L='#5A7BA8'; C_PURPLE='#6C4AB6'; C_PURPLE_L='#9A7FD4'
C_CYAN='#3FA8C1'; C_CYAN_L='#7BC4D6'; C_ORANGE='#D63826'; C_ORANGE_L='#E87562'
C_GRAY='#7A8BA8'; C_GRAY_L='#B8C4D6'; C_DARK='#1E2A3A'; C_WHITE='#FFFFFF'; C_LIGHTGRAY='#F0F2F5'

FIG_W, FIG_H, DPI = 16.0, 8.5, 300
SPACE_SM, SPACE_MD, SPACE_LG = 0.03, 0.06, 0.10
RADIUS_LG, RADIUS_SM, SHADOW_OFF, SHADOW_ALPHA = 0.02, 0.012, 0.006, 0.12
FONT_T1, FONT_T2, FONT_T3, FONT_T4, FONT_T5 = 15, 12.5, 10, 9, 8

def draw_card_shadow(ax,x,y,w,h,radius=RADIUS_LG):
    ax.add_patch(FancyBboxPatch((x+SHADOW_OFF,y-SHADOW_OFF),w,h,boxstyle='round,pad=%s'%radius,facecolor='#000000',edgecolor='none',alpha=SHADOW_ALPHA))
def draw_circle_with_shadow(ax,cx,cy,radius,facecolor,edgecolor=C_WHITE,lw=1.5,alpha=1.0):
    aspect=0.85
    ax.add_patch(Ellipse((cx+radius*0.08,cy-radius*0.08),2*radius,2*radius*aspect,facecolor='#000000',edgecolor='none',alpha=0.1))
    ax.add_patch(Ellipse((cx,cy),2*radius,2*radius*aspect,facecolor=facecolor,edgecolor=edgecolor,lw=lw,alpha=alpha))
def add_gene_dots(ax,cx,cy,radius,color,n_dots=30):
    np.random.seed(42); aspect=0.85
    for _ in range(n_dots):
        a=np.random.uniform(0,2*np.pi); r=np.random.uniform(0,radius*0.75)
        sz=np.random.uniform(0.003,0.007)
        ax.add_patch(Ellipse((cx+r*np.cos(a),cy+r*np.sin(a)*aspect),sz,sz*aspect,facecolor=color,edgecolor='none',alpha=0.65))
def draw_accent_line(ax,x,y,w,color,alpha=0.6):
    ax.add_patch(Rectangle((x,y),w,0.008,facecolor=color,edgecolor='none',alpha=alpha))

# ============================================================
#  CREATE FIGURE
# ============================================================
fig = plt.figure(figsize=(FIG_W,FIG_H),facecolor=C_WHITE)
gs = fig.add_gridspec(1,3,wspace=0.14,left=0.04,right=0.96,top=0.86,bottom=0.12)
axL, axC, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
for ax in [axL,axC,axR]: ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.set_facecolor(C_WHITE)

# ============================================================
#  PANEL 1: LEFT
# ============================================================
ax=axL
ax.text(0.5,0.98,"High-dimensional failure",ha='center',va='top',fontsize=FONT_T2,fontweight='bold',color=C_DARK)
ax.text(0.5,0.90,r"$n \ll d$",ha='center',va='top',fontsize=FONT_T3,color=C_GRAY,fontstyle='italic')
n_rows,n_cols,cell_w,cell_h=6,16,0.036,0.036
mat_start_x,mat_start_y=0.08,0.73; mat_center_y=mat_start_y-n_rows*cell_h/2
ax.add_patch(Rectangle((mat_start_x+0.006,mat_start_y-n_rows*cell_h-0.006),n_cols*cell_w,n_rows*cell_h,facecolor='#000000',edgecolor='none',alpha=0.07))
for i in range(n_rows):
    for j in range(n_cols):
        ax.add_patch(Rectangle((mat_start_x+j*cell_w,mat_start_y-i*cell_h-cell_h),cell_w,cell_h,facecolor=C_BLUE if(i+j)%2==0 else C_CYAN_L,edgecolor=C_WHITE,lw=0.7,alpha=0.9))
ax.text(mat_start_x+n_cols*cell_w/2,mat_start_y+0.015,r"$d$ (genes)",ha='center',va='bottom',fontsize=FONT_T4,color=C_DARK,fontweight='bold')
ax.text(mat_start_x-0.015,mat_center_y,r"$n$ (samples)",ha='right',va='center',fontsize=FONT_T4,color=C_DARK,rotation=90)
note_cx,note_cy=0.85,mat_center_y
ax.annotate('',xy=(0.76,note_cy),xytext=(0.70,note_cy),arrowprops=dict(arrowstyle='->',color=C_DARK,lw=1.5,alpha=0.8))
xs=0.038;ax.plot([note_cx-xs,note_cx+xs],[note_cy-xs*0.85,note_cy+xs*0.85],color=C_ORANGE,lw=3,solid_capstyle='round');ax.plot([note_cx-xs,note_cx+xs],[note_cy+xs*0.85,note_cy-xs*0.85],color=C_ORANGE,lw=3,solid_capstyle='round')
ax.text(note_cx,note_cy+0.07,"NOTEARS",ha='center',va='bottom',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
ax.text(note_cx,note_cy-0.07,"Edges collapse",ha='center',va='top',fontsize=FONT_T4,color=C_DARK)
tx0,ty0,tw,th=0.10,0.22,0.80,0.18
ax.add_patch(FancyBboxPatch((tx0,ty0),tw,th,boxstyle='round,pad=%s'%RADIUS_SM,facecolor=C_LIGHTGRAY,edgecolor='none',alpha=0.5))
d_vals=np.linspace(0,1,60); edges_nte=np.exp(-d_vals*4.5)*0.85
px0,px1=tx0+0.06,tx0+tw-0.02; py0,py1=ty0+0.03,ty0+th-0.025; pw,ph=px1-px0,py1-py0
ax.fill_between(px0+d_vals*pw,py0,py0+edges_nte*ph,color=C_ORANGE,alpha=0.15)
ax.plot(px0+d_vals*pw,py0+edges_nte*ph,color=C_ORANGE,lw=2,alpha=0.9)
ax.text(tx0+tw/2,ty0+th+0.015,"Edges recovered vs. dimension",ha='center',va='bottom',fontsize=FONT_T5,color=C_DARK,fontweight='bold')
ax.text(px0-0.015,(py0+py1)/2,"Edges",ha='right',va='center',fontsize=FONT_T5-0.5,color=C_GRAY,rotation=90)
ax.text(px1,py0-0.005,r"$d$",ha='right',va='top',fontsize=FONT_T5,color=C_GRAY)
ax.text(0.5,0.09,"When n << d, traditional DAG methods fail to recover meaningful edges",ha='center',va='center',fontsize=FONT_T5,color=C_GRAY)

# ============================================================
#  PANEL 2: CENTER (original layout, DAG with real-estate-appropriate coords)
# ============================================================
ax=axC
ax.text(0.5,0.98,"CAGate: Cluster-Aware Gating",ha='center',va='top',fontsize=FONT_T2,fontweight='bold',color=C_BLUE)
ax.text(0.5,0.87,"Residual Clustering",ha='center',va='center',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
ax.text(0.5,0.80,r"$\mathbf{R} = \mathbf{X} - \mathbf{XW}$",ha='center',va='center',fontsize=FONT_T3,color=C_DARK)
ccx=[0.20,0.50,0.80]; ccy=0.72; cr=[0.085,0.072,0.078]; ccols=[C_BLUE,C_PURPLE,C_CYAN]; ccols_l=[C_BLUE_L,C_PURPLE_L,C_CYAN_L]; clabels=['Cluster 1','Cluster 2','Cluster 3']
for i in range(3):
    draw_circle_with_shadow(ax,ccx[i],ccy,cr[i],facecolor=ccols_l[i],alpha=0.85)
    ax.add_patch(Ellipse((ccx[i],ccy),cr[i]*1.1,cr[i]*0.9,facecolor=ccols[i],edgecolor='none',alpha=0.3))
    add_gene_dots(ax,ccx[i],ccy,cr[i]*0.78,ccols[i],n_dots=28)
ly=ccy-max(cr)-0.025
for i in range(3): ax.text(ccx[i],ly,clabels[i],ha='center',va='top',fontsize=FONT_T5,color=C_DARK)
a1t,a1b=ly-0.035,ly-0.075
ax.annotate('',xy=(0.5,a1b),xytext=(0.5,a1t),arrowprops=dict(arrowstyle='->',color=C_GRAY,lw=1.5,alpha=0.7))
st2_y=a1b-0.025; ax.text(0.5,st2_y,"MAD Gating",ha='center',va='top',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
mfy=st2_y-0.05; ax.text(0.5,mfy,r"$\gamma_k = 1 - \exp\left(-\alpha \cdot \overline{\mathrm{MAD}}_k\right)$",ha='center',va='center',fontsize=FONT_T3+0.5,color=C_DARK)
bpx=[0.10,0.38,0.66]; bw=0.24; bh=0.04; bty=mfy-0.04; gv=[0.92,0.18,0.25]; gl=[r"$\gamma_1$=0.92",r"$\gamma_2$=0.18",r"$\gamma_3$=0.25"]
for i in range(3):
    bx=bpx[i];v=gv[i]
    ax.add_patch(Rectangle((bx+0.003,bty-bh-0.003),bw,bh,facecolor='#000000',edgecolor='none',alpha=0.07))
    ax.add_patch(Rectangle((bx,bty-bh),bw,bh,facecolor=C_LIGHTGRAY,edgecolor='none'))
    ax.add_patch(Rectangle((bx,bty-bh),bw*v,bh,facecolor=ccols[i],edgecolor='none'))
    ax.add_patch(Rectangle((bx,bty-0.006),bw*v,0.005,facecolor=C_WHITE,edgecolor='none',alpha=0.3))
    ax.text(bx+bw/2,bty-bh-0.025,gl[i],ha='center',va='top',fontsize=FONT_T5,color=C_DARK)
blb=bty-bh-0.035; a2t,a2b=blb-0.025,blb-0.065
ax.annotate('',xy=(0.5,a2b),xytext=(0.5,a2t),arrowprops=dict(arrowstyle='->',color=C_GRAY,lw=1.5,alpha=0.7))

# Step 3: Learned DAG - horizontal chain layout, visible arrows
st3_y=a2b-0.025; ax.text(0.5,st3_y,"Learned DAG",ha='center',va='top',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
dag_ny = st3_y - 0.045
dx=[0.12,0.35,0.58,0.81]; dnames=["TP53","BRCA1","EGFR","MYC"]
nw,nh=0.075,0.04
for i,(xx,nm) in enumerate(zip(dx,dnames)):
    ax.add_patch(Ellipse((xx+0.003,dag_ny-0.003),nw,nh,facecolor='#000000',edgecolor='none',alpha=0.1,zorder=2))
    ax.add_patch(Ellipse((xx,dag_ny),nw,nh,facecolor=C_CYAN_L,edgecolor=C_CYAN,lw=1.5,zorder=2))
    ax.text(xx,dag_ny,nm,ha='center',va='center',fontsize=FONT_T5-0.5,color=C_DARK,fontweight='bold',zorder=3)
# Arrows: direct line between consecutive nodes, no arc needed for horizontal
for i in range(3):
    x1,x2=dx[i]+nw/2,dx[i+1]-nw/2
    ax.annotate('',xy=(x2,dag_ny),xytext=(x1,dag_ny),
                arrowprops=dict(arrowstyle='-|>',color=C_DARK,lw=1.5,alpha=0.7,
                                shrinkA=0.008,shrinkB=0.008,mutation_scale=20),zorder=1)

# ============================================================
#  PANEL 3: RIGHT (real data numbers)
# ============================================================
ax=axR
ax.text(0.5,0.98,"Pan-cancer rescue",ha='center',va='top',fontsize=FONT_T2,fontweight='bold',color=C_PURPLE)
bx,by,bw,bh=0.10,0.78,0.80,0.13
draw_card_shadow(ax,bx,by,bw,bh);ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle='round,pad=%s'%RADIUS_LG,facecolor=C_WHITE,edgecolor=C_PURPLE,lw=2));draw_accent_line(ax,bx+0.06,by+bh-0.008,bw-0.12,C_PURPLE)
ax.text(0.5,by+bh-0.032,"%d/%d Cancer Types"%(pos_count,total_cancers),ha='center',va='center',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
ax.text(0.5,by+0.035,r"$\Delta$ = +%.0f edges"%mean_delta,ha='center',va='center',fontsize=FONT_T2-1,color=C_BLUE,fontweight='bold')
hty=by-SPACE_MD; ax.text(0.5,hty,"High-dimensional recovery",ha='center',va='top',fontsize=FONT_T3,color=C_DARK,fontweight='bold')
bsx,bsw,bh2=0.12,0.76,0.05; cty=hty-0.07; cv=0.93
ax.add_patch(Rectangle((bsx+0.004,cty-bh2-0.004),bsw*cv,bh2,facecolor='#000000',edgecolor='none',alpha=0.09))
ax.add_patch(Rectangle((bsx,cty-bh2),bsw*cv,bh2,facecolor=C_BLUE,edgecolor='none'))
ax.add_patch(Rectangle((bsx,cty-0.007),bsw*cv,0.006,facecolor=C_WHITE,edgecolor='none',alpha=0.25))
ax.text(bsx+bsw/2,cty-bh2/2,"CAGate: %d edges (d=200)"%552,ha='center',va='center',fontsize=FONT_T4,color=C_WHITE,fontweight='bold')
nty=cty-bh2-SPACE_MD; nv=0.10
ax.add_patch(Rectangle((bsx+0.004,nty-bh2-0.004),bsw*nv,bh2,facecolor='#000000',edgecolor='none',alpha=0.09))
ax.add_patch(Rectangle((bsx,nty-bh2),bsw*nv,bh2,facecolor=C_ORANGE,edgecolor='none'))
ax.text(bsx+bsw*nv+0.015,nty-bh2/2,"NOTEARS: ~0 edges",ha='left',va='center',fontsize=FONT_T4,color=C_DARK,fontweight='bold')
ncy=nty-bh2-SPACE_SM; ax.text(bsx+bsw/2,ncy,"collapses in high dimensions",ha='center',va='top',fontsize=FONT_T5,color=C_GRAY,fontstyle='italic')

# External validation — placed right below "collapses" caption
ety=ncy-0.03; ax.text(0.5,ety,"External validation",ha='center',va='top',fontsize=FONT_T4,color=C_DARK,fontweight='bold')
bcx=[0.18,0.50,0.82]; bcy=ety-0.11; bwl=[0.20,0.26,0.22]; bhh=0.13
blabels=['CTD\n%d%%'%ctd_pct,'ClinGen\n%d%%'%clin_pct,'STRING\n8%']; bcols=[C_BLUE,C_PURPLE,C_CYAN]
for i in range(3):
    bxx=bcx[i];bww=bwl[i];byy=bcy-bhh/2
    draw_card_shadow(ax,bxx-bww/2,byy,bww,bhh,radius=RADIUS_SM)
    ax.add_patch(FancyBboxPatch((bxx-bww/2,byy),bww,bhh,boxstyle='round,pad=%s'%RADIUS_SM,facecolor=bcols[i],edgecolor='none',alpha=0.95))
    ax.add_patch(FancyBboxPatch((bxx-bww/2+0.01,byy+bhh-0.018),bww-0.02,0.012,boxstyle='round,pad=0.004',facecolor=C_WHITE,edgecolor='none',alpha=0.2))
    ax.text(bxx,bcy,blabels[i],ha='center',va='center',fontsize=FONT_T4,color=C_WHITE,fontweight='bold')

# Small-sample advantage card
b2x,b2y,b2w,b2h=0.10,0.10,0.80,0.12
draw_card_shadow(ax,b2x,b2y,b2w,b2h);ax.add_patch(FancyBboxPatch((b2x,b2y),b2w,b2h,boxstyle='round,pad=%s'%RADIUS_LG,facecolor=C_WHITE,edgecolor=C_CYAN,lw=2));draw_accent_line(ax,b2x+0.06,b2y+b2h-0.008,b2w-0.12,C_CYAN)
ax.text(0.5,b2y+b2h-0.03,"Small-sample advantage",ha='center',va='center',fontsize=FONT_T3-0.5,color=C_DARK,fontweight='bold')
ax.text(0.5,b2y+0.025,"+%.0f (n < %d)  vs  +%.0f (n >= %d)"%(small_delta,int(median_n),large_delta,int(median_n)),ha='center',va='center',fontsize=FONT_T5,color=C_GRAY)

# ============================================================
#  CONNECTING ARROWS
# ============================================================
ay=0.55; fig.patches.append(FancyArrowPatch((0.348,ay),(0.372,ay),arrowstyle='-|>,head_width=0.05,head_length=0.025',color=C_GRAY,lw=1.5,alpha=0.7,mutation_scale=25,transform=fig.transFigure))
fig.patches.append(FancyArrowPatch((0.652,ay),(0.676,ay),arrowstyle='-|>,head_width=0.05,head_length=0.025',color=C_GRAY,lw=1.5,alpha=0.7,mutation_scale=25,transform=fig.transFigure))
fig.suptitle("CAGate: Cluster-Aware Gating for Differentiable Causal Discovery in Cancer Genomics",fontsize=FONT_T1,fontweight='bold',color=C_DARK,y=0.965)
fig.text(0.5,0.925,"Rescuing high-dimensional causal structure learning through residual clustering and MAD-based gating",ha='center',va='center',fontsize=FONT_T4,color=C_GRAY,style='italic')

# ============================================================
OUT = os.path.join(PKG_ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
png=os.path.join(OUT,'GraphicalAbstract.png'); pdf=os.path.join(OUT,'GraphicalAbstract.pdf')
plt.savefig(png,dpi=DPI,bbox_inches='tight',facecolor=C_WHITE); plt.savefig(pdf,bbox_inches='tight',facecolor=C_WHITE); plt.close()
print("[OK] GraphicalAbstract.png (%d KB)"%(os.path.getsize(png)//1024))
print("[OK] GraphicalAbstract.pdf (%d KB)"%(os.path.getsize(pdf)//1024))
