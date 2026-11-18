"""Project Proteus v3 PDF report: presentation only."""
from pathlib import Path
import textwrap, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
NAVY, AMBER, MUTED = "#0b1626", "#e8890c", "#64748b"
ATTRIBUTION = ("© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium. "
    "WorldCover 2021 v200, CC BY 4.0. DOI: 10.5281/zenodo.7254221. Sentinel data are subject to the Copernicus Sentinel legal notice; other missions retain their own licences.")
DISCLAIMER="Suspected oil only. Not a confirmed spill. Independent validation has not been completed."
CAPTIONS=[("oil_preview.png","Spatial screening preview","Downsampled display: SAR intensity greyscale, water blue, suspected oil amber, excluded/unknown grey."),
    ("backscatter_histogram.png","Backscatter distribution","Backscatter distributions and the effective seed threshold support review; they are not independently validated oil-discrimination criteria."),
    ("object_statistics.png","Object statistics","Candidate area and contrast support review. Heuristic confidence and model scores are not calibrated probabilities.")]
def fmt(value,d=3):
    try:
        v=float(value)
        return f"{v:,.{d}f}" if math.isfinite(v) else "Unavailable"
    except (TypeError,ValueError): return "Unavailable"
def lines(value,width=82):
    return [line for para in str(value if value is not None else "Unavailable").splitlines() for line in (textwrap.wrap(para,width=width,break_long_words=True) or [""])] or [""]
class Layout:
    def __init__(self,pdf): self.pdf=pdf; self.fig=None; self.page=0; self.y=.85
    def text(self,x,y,t,size=9,bold=False,color=NAVY):
        self.fig.text(x,y,str(t),fontsize=size,va="top",color=color,weight="bold" if bold else "normal",family="DejaVu Sans")
    def flush(self):
        if self.fig is not None: self.pdf.savefig(self.fig); plt.close(self.fig); self.fig=None
    def new(self,title):
        self.flush(); self.page+=1; self.fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
        self.text(.07,.965,"Project Proteus v3",10,True); self.text(.07,.925,title,17,True)
        self.text(.07,.889,"Analyst-review screening product",8,color=MUTED)
        self.fig.add_artist(plt.Line2D([.07,.93],[.868,.868],transform=self.fig.transFigure,color=AMBER,linewidth=1.6))
        self.fig.add_artist(plt.Line2D([.07,.93],[.06,.06],transform=self.fig.transFigure,color="#dce3eb",linewidth=.6))
        self.text(.07,.044,"Suspected oil only · Independent validation not completed",7,color=MUTED)
        self.text(.88,.044,f"{self.page:02d}",8,color=MUTED); self.y=.842
    def ensure(self,h):
        if self.y-h<.085: self.new("Report continued")
    def paragraph(self,v,size=9):
        for line in lines(v,82):
            self.ensure(.019); self.text(.07,self.y,line,size); self.y-=.019
        self.y-=.012
    def heading(self,t):
        self.ensure(.075); self.y-=.012; self.text(.07,self.y,t,11,True); self.y-=.035
    def pair(self,k,v):
        ll=lines(k,22); rr=lines(v,52); count=max(len(ll),len(rr))
        for start in range(0,count,22):
            left=ll[start:start+22]; right=rr[start:start+22]; h=max(len(left),len(right))*.019+.018
            self.ensure(h)
            for i,t in enumerate(left): self.text(.07,self.y-i*.019,t,8.5,True)
            for i,t in enumerate(right): self.text(.34,self.y-i*.019,t,9)
            self.y-=h
            self.fig.add_artist(plt.Line2D([.07,.93],[self.y+.007]*2,transform=self.fig.transFigure,color="#e2e8f0",linewidth=.5))
def build_pdf(results,report,rows,meta):
    results=Path(results); results.mkdir(parents=True,exist_ok=True)
    screen=meta.get("screening",{}); obj=report.get("object_summary",{}); area=report.get("estimated_oil_area_km2")
    with PdfPages(results/"analysis_report.pdf") as pdf:
        pdf.infodict().update(Title="Project Proteus v3 — Analysis Report",Subject="Suspected-oil screening; not spill confirmation")
        d=Layout(pdf); d.new("Executive screening summary"); d.paragraph(DISCLAIMER,8.5)
        for k,v in [("Scene",report.get("source_filename")),("Acquisition",meta.get("acquisition") or "Unavailable"),
            ("Screening level",screen.get("severity","Unavailable")),("Suspected area",fmt(area,4)+" km²" if area is not None else "Unavailable for this CRS geometry"),
            ("Affected water",fmt(screen.get("affected_water_pct"),4)+"% of evaluated water"),("Candidate objects",obj.get("num_objects",0)),
            ("Confidence counts","High {0}; medium {1}; low {2}; unknown {3}".format(*[obj.get("num_"+c+"_confidence",0) for c in ["high","medium","low","unknown"]])),
            ("Review flags","; ".join(screen.get("flags",[])) or "None raised; this does not establish absence of oil."),
            ("Device",meta.get("device_note") or report.get("device","Unavailable")),
            ("Runtime",str(report.get("processing",{}).get("runtime_seconds",meta.get("runtime_s","Unavailable")))+" s")]: d.pair(k,v)
        d.heading("Interpretation"); d.paragraph("Candidates require analyst review and independent context. Low wind, biogenic films and rain can resemble oil. Screening level is a heuristic review priority, not environmental severity or a regulatory threshold.")
        d.heading("Essential limitations"); d.paragraph("Training uses physics-generated pseudo-labels, not verified spill labels. Model scores and confidence classes are not calibrated probabilities. Low-contrast oil may be missed. Do not use this report alone for spill confirmation or attribution.",8.5)
        d.new("Scene and processing record")
        for k,v in [("Scene",report.get("source_filename")),("Bounds (WGS84)",meta.get("bounds")),("Run (UTC)",meta.get("run_time_utc")),
            ("Internal trace ID",meta.get("run_id")),("Processing",report.get("processing",{}).get("strategy","Unavailable")),("Device",meta.get("device_note"))]: d.pair(k,v)
        d.heading("Screening settings"); st=meta.get("settings",{})
        for k,v in [("Sensitivity",report.get("mode")),("Base threshold (dB)",st.get("base_threshold",-25)),("Effective seed (dB)",report.get("threshold_db")),
            ("Minimum local contrast",report.get("contrast_min")),("Growth allowance (dB)",report.get("growth_allowance_db")),("Model score threshold",report.get("model_threshold"))]: d.pair(k,v)
        d.new("Reference and software record"); d.heading("WorldCover tiles")
        refs=report.get("reference_files",[]) or report.get("reference_sources",[])
        if refs:
            for r in refs: d.paragraph(r,8)
        else: d.paragraph("See the JSON reference manifest for this run.")
        d.heading("Software versions")
        for k,v in meta.get("packages",{}).items(): d.pair(k,v)
        for i,(name,title,cap) in enumerate([("oil_preview.png","Spatial screening preview","Downsampled display: SAR intensity greyscale, water blue, suspected oil amber, excluded/unknown grey."),("backscatter_histogram.png","Backscatter distribution","Backscatter distributions and the effective seed threshold support review; they are not independently validated oil-discrimination criteria."),("object_statistics.png","Object statistics","Candidate area and contrast support review. Heuristic confidence and model scores are not calibrated probabilities.")],1):
            p=results/name
            if not p.is_file(): continue
            d.new(f"Figure {i}: {title}")
            ax=d.fig.add_axes([.07,.23,.86,.59]); ax.axis("off"); ax.imshow(mpimg.imread(p),aspect="equal")
            d.y=.19; d.paragraph(cap,8.5)
        d.new("Ten largest candidate objects")
        d.paragraph("Ranked by estimated area where available. Connected objects are not necessarily separate spills. Confidence values are heuristic, not probabilities.",8.5)
        top=sorted(rows,key=lambda r:-float(r.get("area_km2") or 0))[:10]
        if top:
            headers=["Object","Area\n(km²)","Contrast\n(dB)","Model\nscore","Confidence\nindex","Class"]
            data=[[str(r.get("object_id","")),fmt(r.get("area_km2"),4),fmt(r.get("contrast_vs_water_db"),2),fmt(r.get("mean_model_score"),3),fmt(r.get("heuristic_confidence"),3),str(r.get("confidence_class","unknown"))] for r in top]
            ax=d.fig.add_axes([.07,.29,.86,.42]); ax.axis("off")
            t=ax.table(cellText=data,colLabels=headers,cellLoc="center",colLoc="center",colWidths=[.10,.19,.18,.18,.18,.17],loc="upper center")
            t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1,1.9)
            for (r,c),cell in t.get_celld().items():
                cell.set_edgecolor("#dce3eb"); cell.set_linewidth(.5)
                cell.set_facecolor(NAVY if r==0 else ("#f1f5f9" if r%2 else "white"))
                if r==0: cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        else: d.paragraph("No candidate objects detected at the selected settings. This does not confirm oil-free water.")
        d.new("Limitations and attribution"); d.heading("Interpretation limits")
        lim=list(report.get("limitations",[]))+["Look-alikes may be selected; low-contrast oil may be missed.","Areas depend on CRS geometry and reference age/coverage. WorldCover 2021 is not a contemporaneous coastline survey.","Confidence indices and model scores are heuristic, not calibrated spill probabilities.","Training loss against physics-generated labels is not independent detection accuracy.","No vessel attribution or environmental-harm quantification is provided."]
        for item in dict.fromkeys(lim): d.paragraph("• "+str(item),8.5)
        d.heading("Data attribution"); d.paragraph(ATTRIBUTION,8); d.flush()
    return results/"analysis_report.pdf"
