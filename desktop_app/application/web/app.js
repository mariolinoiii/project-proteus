"use strict";
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(v,d=2)=>v===""||v==null||isNaN(+v)?"–":(+v).toLocaleString(undefined,{maximumFractionDigits:d});
let ST={},DIAG=null,BUSY=false;const PICK=new Set();
const seg={aRef:"local",dRef:"local",aDev:"Auto",tDev:"Auto"};
const TITLES={analysis:["Analysis Workspace","Locate candidate dark features and prioritise them for analyst review."],training:["Model Training","Train ProteusSegNet on the prepared dataset; the result is used automatically."],dataset:["Dataset Preparation","Build SAR physics pseudo-labels and review the WorldCover reference."],project:["Project Proteus","Project Proteus, the ProteusSegNet application and its limits."]};

async function api(p,o){const r=await fetch(p,o);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.statusText);return j}
const post=(p,b)=>api(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
function showErr(m){
if(m){const msg=String(m);const tips=[[/Failed to fetch|NetworkError|Upload failed/i,"Keep the Proteus console open, check the local connection and retry."],[/CUDA|GPU mode requires/i,"Select Auto or CPU if a working CUDA device is unavailable."],[/WorldCover|reference.*available|reference-water/i,"Upload the required WorldCover tiles or choose Online; unknown coverage stays excluded."],[/No trained model|checkpoint.*missing/i,"Prepare a dataset and train, or select a trusted compatible checkpoint."],[/CRS|single.band|backscatter/i,"Use a calibrated single-band dB GeoTIFF with a CRS."],[/disk space/i,"Free storage or move the workspace/scratch folder to a larger drive."]];const hit=tips.find(([p])=>p.test(msg));m=hit?msg+"\nNext step: "+hit[1]:msg;}
const e=$("#err");e.textContent=m;e.classList.toggle("hidden",!m)}

function fmtT(s){s=Math.round(s);return Math.floor(s/60)+":"+String(s%60).padStart(2,"0")}
function jobUI(j){const b=$("#jobbar");b.classList.remove("hidden");b.classList.toggle("done",j.status=="done");b.classList.toggle("error",j.status=="error");
 const stalled=j.status=="running"&&j.heartbeat&&(Date.now()/1000-j.heartbeat>45);b.classList.toggle("stalled",stalled);
 $("#jkind").textContent=j.kind||"";$("#jmsg").textContent=j.status=="error"?"Failed: "+(j.error||""):(stalled?(j.message||"")+" (no activity for 45 s; large steps can be silent)":j.message||"Working");
 $("#jpct").textContent=Math.round((j.progress||0)*100)+"%";$("#jfill").style.width=(j.progress||0)*100+"%";$("#jtime").textContent=fmtT(j.elapsed||0);
 $("#jdev").textContent=j.device||"";const l=$("#jlog"),atEnd=l.scrollTop+l.clientHeight>=l.scrollHeight-8;l.textContent=(j.log||[]).slice(-60).join("\n");if(atEnd)l.scrollTop=l.scrollHeight}
$("#jclose").onclick=()=>$("#jobbar").classList.add("hidden");
function upload(kind,file){return new Promise((res,rej)=>{const x=new XMLHttpRequest(),t0=Date.now();x.open("POST",`/api/upload?kind=${kind}&name=${encodeURIComponent(file.name)}`);BUSY=true;
 x.upload.onprogress=e=>{if(e.lengthComputable)jobUI({kind:"upload",status:"running",message:`Uploading ${file.name} (${fmt(e.loaded/1048576,0)} / ${fmt(e.total/1048576,0)} MB)`,progress:e.loaded/e.total,elapsed:(Date.now()-t0)/1000,heartbeat:Date.now()/1000,log:[`Uploading ${file.name}`]})};
 x.onload=()=>{BUSY=false;let j={};try{j=JSON.parse(x.responseText)}catch(e){}if(x.status==200){jobUI({kind:"upload",status:"done",message:`Uploaded ${file.name}`,progress:1,elapsed:(Date.now()-t0)/1000,log:[`Uploaded ${file.name}`]});res(j.name)}else rej(new Error(j.error||"Upload failed"))};
 x.onerror=()=>{BUSY=false;rej(new Error("Upload failed"))};x.send(file)})}
function runJob(path,body){showErr("");BUSY=true;return post(path,body).then(({job})=>new Promise((res,rej)=>{const t=setInterval(async()=>{try{const j=await api("/api/job/"+job);jobUI(j);
 if(j.status!="running"){clearInterval(t);BUSY=false;j.status=="done"?res(j.result):rej(new Error(j.error))}}catch(e){clearInterval(t);BUSY=false;rej(e)}},700)})).catch(e=>{BUSY=false;showErr(e.message);throw e})}

function setView(v){$$("#nav button").forEach(b=>b.classList.toggle("on",b.dataset.view==v));$$(".view").forEach(s=>s.classList.toggle("on",s.id=="v-"+v));
 $("#title").textContent=TITLES[v][0];$("#sub").textContent=TITLES[v][1];if(v=="project")loadDiag();if(v=="dataset")loadDataset();if(v=="training")renderTraining()}
$("#nav").onclick=e=>{const b=e.target.closest("button");if(b)setView(b.dataset.view)};
$("#tGo").onclick=()=>setView("dataset");
$$(".seg").forEach(s=>s.onclick=e=>{const b=e.target.closest("button");if(!b)return;s.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x==b));seg[s.id]=b.dataset.v;segChanged(s.id)});
function segChanged(id){if(id=="aRef"){$("#aRefLocal").classList.toggle("hidden",seg.aRef!="local");$("#aRefOnline").classList.toggle("hidden",seg.aRef!="online")}
 if(id=="dRef"){$("#dRefLocal").classList.toggle("hidden",seg.dRef!="local");$("#dRefOnline").classList.toggle("hidden",seg.dRef!="online")}devNote()}
function devNote(){const n=!DIAG?"":DIAG.cuda_ok?`GPU ready: ${DIAG.gpu}`:"No working CUDA device. Auto will use the CPU; GPU mode will stop with an error.";["aDevNote","tDevNote"].forEach(i=>$("#"+i).textContent=n)}
function fill(sel,items,first){const el=$(sel),cur=el.value;el.innerHTML=(first!=null?`<option value="">${esc(first)}</option>`:"")+items.map(i=>`<option value="${esc(i)}">${esc(i)}</option>`).join("");if([...el.options].some(o=>o.value==cur))el.value=cur}

function renderScenes(){const list=ST.uploads.sar;[...PICK].forEach(n=>{if(!list.includes(n))PICK.delete(n)});
 $("#dScenes").innerHTML=list.map(x=>`<label><input type="checkbox" value="${esc(x)}" ${PICK.has(x)?"checked":""}> ${esc(x)}</label>`).join("")||'<span class="muted small">Upload SAR scenes to begin.</span>';
 $("#dCount").textContent=list.length?`${PICK.size} of ${list.length} selected`:"";$("#dRun").disabled=PICK.size==0}
$("#dScenes").onchange=e=>{if(e.target.matches("input")){e.target.checked?PICK.add(e.target.value):PICK.delete(e.target.value);renderScenes()}};
$("#dAll").onclick=()=>{ST.uploads.sar.forEach(n=>PICK.add(n));renderScenes()};$("#dNone").onclick=()=>{PICK.clear();renderScenes()};

async function refresh(){ST=await api("/api/state");
 fill("#aSar",ST.uploads.sar,"Select an uploaded scene");fill("#aOverride",ST.uploads.models,"Default trained model");
 const ck=ST.checkpoint;$("#chipModel").textContent=ck.ready?"Model ready":"No trained model";$("#chipModel").className="chip "+(ck.ready?"ok":"warn");
 $("#aModel").innerHTML=ck.ready?`<span class="good">Trained model ready</span> · updated ${esc(ck.modified)} · ${ck.size_mb} MB. It is used automatically.`:"No trained model yet. Train one in Model Training, or load your own checkpoint under Advanced.";
 $("#chipDs").textContent=ST.dataset?"Dataset ready":"No dataset";$("#chipDs").className="chip "+(ST.dataset?"ok":"");
 const n=ST.worldcover_cache.length;["aRefInfo","dRefInfo"].forEach(i=>$("#"+i).textContent=n?`${n} cached tile(s) available`:"No cached tiles");
 renderScenes();
 try{DIAG=await api("/api/diagnostics");$("#chipDev").textContent=DIAG.core_error?"Setup incomplete":(DIAG.cuda_ok?"GPU: "+DIAG.gpu:"CPU only");$("#chipDev").className="chip "+(DIAG.core_error?"warn":"ok");devNote()}catch(e){}
 if(ST.job&&!BUSY){resume(ST.job)}}
function resume(id){BUSY=true;const t=setInterval(async()=>{try{const j=await api("/api/job/"+id);jobUI(j);if(j.status!="running"){clearInterval(t);BUSY=false}}catch(e){clearInterval(t);BUSY=false}},900)}

function effective(){const off={Conservative:-2,Balanced:0,Sensitive:2}[$("#aMode").value],b=+$("#aBase").value;$("#aEff").textContent=`Base ${b} dB ${off<0?"−":"+"} offset ${Math.abs(off)} dB = effective seed ${b+off} dB`;$("#aBaseO").textContent=b+" dB";$("#aMtO").textContent=(+$("#aMt").value).toFixed(2)}
["aMode","aBase","aMt"].forEach(i=>$("#"+i).addEventListener("input",effective));
$("#dBase").oninput=()=>$("#dBaseO").textContent=$("#dBase").value+" dB";$("#tEp").oninput=()=>$("#tEpO").textContent=$("#tEp").value;

async function inspect(){const n=$("#aSar").value,box=$("#aInspect");if(!n){box.innerHTML="Choose a calibrated, single-band dB GeoTIFF with a CRS.";return}
 box.textContent="Inspecting…";try{const i=await api("/api/inspect?name="+encodeURIComponent(n));
 box.innerHTML=`${i.width.toLocaleString()} × ${i.height.toLocaleString()} px · ${i.bands} band · ${esc(i.crs||"no CRS")}<br>Plan: ${esc(i.plan)}`+(i.stats?`<br>dB sample: p1 ${fmt(i.stats.p1,1)}, median ${fmt(i.stats.median,1)}, p99 ${fmt(i.stats.p99,1)}`:"")
 +`<br>WorldCover tiles: ${i.cached_tiles.length}/${i.tiles.length} cached`+(i.missing_tiles.length?` <span class="warn">(missing ${i.missing_tiles.map(esc).join(", ")})</span>`:` <span class="good">✓</span>`)
 +`<br>Acquisition: ${esc(i.acquisition||"Unavailable")}`+i.issues.map(x=>`<br><span class="bad">✕ ${esc(x)}</span>`).join("")+i.notes.map(x=>`<br><span class="warn">! ${esc(x)}</span>`).join("")}catch(e){box.innerHTML=`<span class="bad">${esc(e.message)}</span>`}}
$("#aSar").onchange=inspect;
async function ups(input,kind,onName){for(const f of input.files){try{const n=await upload(kind,f);onName&&onName(n)}catch(e){showErr(e.message)}}input.value="";await refresh()}
$("#aSarUp").onchange=async e=>{let last="";await ups(e.target,"sar",n=>last=n);if(last){$("#aSar").value=last;inspect()}};
$("#aRefUp").onchange=async e=>{await ups(e.target,"reference");inspect()};$("#dRefUp").onchange=e=>ups(e.target,"reference");
$("#dSarUp").onchange=e=>ups(e.target,"sar",n=>PICK.add(n));
$("#aModelUp").onchange=async e=>{let last="";await ups(e.target,"model",n=>last=n);if(last)$("#aOverride").value=last};

$("#aRun").onclick=async()=>{if(!$("#aSar").value)return showErr("Select a SAR scene first.");$("#aRun").disabled=true;
 try{showAnalysis(await runJob("/api/analysis/run",{sar:$("#aSar").value,reference_mode:seg.aRef,checkpoint_override:$("#aOverride").value,mode:$("#aMode").value,base_threshold:+$("#aBase").value,model_threshold:+$("#aMt").value,zero_exclusion:$("#aZero").checked,device:seg.aDev,strategy:$("#aStrat").value,tile_size:+$("#aTile").value}))}catch(e){}$("#aRun").disabled=false;refresh()};
function showAnalysis(r){const rep=r.report,s=r.meta.screening,a=rep.estimated_oil_area_km2,o=rep.object_summary;$("#aOut").classList.remove("hidden");
 const k=(l,v,c="")=>`<div class="kpi ${c}"><small>${l}</small><b>${v}</b></div>`;
 $("#aKpi").innerHTML=k("Screening level",`<span class="sev-${esc(s.severity.split(" ")[0])}">${esc(s.severity)}</span>`)+k("Suspected area",a==null?"n/a":fmt(a,3)+" km²","amb")+k("Objects",o.num_objects)+k("High / medium / low",`${o.num_high_confidence} / ${o.num_medium_confidence} / ${o.num_low_confidence}`)+k("Affected water",fmt(s.affected_water_pct,3)+"%")+k("Device used",esc(rep.device.toUpperCase()))+k("Effective seed",fmt(rep.threshold_db,1)+" dB")+k("Runtime",fmt(r.meta.runtime_s,1)+" s");
 $("#aFigs").innerHTML=r.figures.map((f,i)=>`<img src="${f}?t=${Date.now()}" alt="Figure ${i+1}">`).join("");
 const cols=["object_id","area_km2","contrast_vs_water_db","mean_backscatter_db","mean_model_score","heuristic_confidence","confidence_class"];
 $("#aTab").innerHTML="<tr>"+cols.map(c=>`<th>${c.replace(/_/g," ")}</th>`).join("")+"</tr>"+(r.objects.length?r.objects.map(x=>"<tr>"+cols.map(c=>`<td>${esc(isNaN(+x[c])||x[c]===""?x[c]:fmt(x[c],4))}</td>`).join("")+"</tr>").join(""):`<tr><td colspan="7">No objects detected.</td></tr>`);
 $("#dlZip").href=`/api/runs/${encodeURIComponent(r.run_id)}/zip`;$("#dlPdf").href=`/api/runs/${encodeURIComponent(r.run_id)}/pdf`;$("#aOut").scrollIntoView({behavior:"smooth"})}

$("#dRun").onclick=async()=>{const scenes=[...PICK];if(!scenes.length)return showErr("Tick at least one SAR scene in the list.");$("#dRun").disabled=true;
 try{await runJob("/api/dataset/prepare",{scenes,reference_mode:seg.dRef,mode:$("#dMode").value,base_threshold:+$("#dBase").value,zero_exclusion:$("#dZero").checked,strategy:$("#dStrat").value,tile_size:+$("#dTile").value});await refresh();await loadDataset(true)}catch(e){}renderScenes()};
async function loadDataset(force){try{const d=await api("/api/dataset/scenes");if(!d.dataset){$("#dOut").classList.add("hidden");return}
 $("#dOut").classList.remove("hidden");const sel=$("#dSel");if(force||sel.options.length!=d.scenes.length)sel.innerHTML=d.scenes.map(s=>`<option>${esc(s.image_id)}</option>`).join("");await scene()}catch(e){showErr(e.message)}}
$("#dSel").onchange=scene;
async function scene(){const id=$("#dSel").value;if(!id)return;$("#dFigs").style.opacity=.4;try{const r=await api("/api/dataset/preview?scene="+encodeURIComponent(id)),u=r.urls,s=r.stats,t=s.evaluated_water_px+s.oil_px+s.excluded_px||1;
 $("#dFigs").innerHTML=["sar","worldcover","labels","overlay"].map(k=>`<img src="${u[k]}" alt="${k}">`).join("");
 $("#dCharts").innerHTML=`<img src="${u.worldcover_chart}" alt="WorldCover graphs"><img src="${u.label_chart}" alt="Labelling graphs">`;
 const k=(l,v,c="")=>`<div class="kpi ${c}"><small>${l}</small><b>${v}</b></div>`;
 $("#dKpi").innerHTML=k("Evaluated water",fmt(100*s.evaluated_water_px/t,2)+"%")+k("Suspected oil pixels",s.oil_px.toLocaleString(),"amb")+k("Excluded",fmt(100*s.excluded_px/t,2)+"%")+k("WorldCover water",s.wc_water_pct+"%")}catch(e){showErr(e.message)}$("#dFigs").style.opacity=1}

function renderTraining(){const d=ST.dataset;$("#tDs").innerHTML=d?`Latest dataset ready (${esc(d.created)}). Sensitivity ${esc(d.settings.mode)}, base threshold ${d.settings.base_threshold} dB.`:"No dataset prepared yet. Prepare one in Dataset Preparation first.";
 $("#tGo").classList.toggle("hidden",!!d);
 $("#tCk").innerHTML=ST.checkpoint.ready?`Trained model ready · updated ${esc(ST.checkpoint.modified)}. Training replaces it.`:"No trained model yet.";$("#tRun").disabled=!d}
$("#tRun").onclick=async()=>{$("#tRun").disabled=true;try{const r=await runJob("/api/train/start",{epochs:+$("#tEp").value,device:seg.tDev});$("#tOut").classList.remove("hidden");$("#tFig").src=r.figure+"?t="+Date.now();
 const k=(l,v)=>`<div class="kpi"><small>${l}</small><b>${v}</b></div>`;
 $("#tKpi").innerHTML=k("Final loss",fmt(r.final_loss,4))+k("First-epoch loss",fmt(r.first_loss,4))+k("Epochs",r.epochs)+k("Device",esc((r.device_note||"").replace(/^(Auto: |GPU selected: |CPU selected\.?)/,"")||"CPU"))+k("Finished",esc((r.finished||"").replace("T"," ")));await refresh()}catch(e){}renderTraining()};

async function loadDiag(){try{const d=await api("/api/diagnostics");$("#diag").textContent=`Python ${d.python}\nPyTorch ${d.torch||"n/a"} (CUDA build ${d.cuda_build||"none"})\nDetected GPU: ${d.gpu||"none"}\nNVIDIA driver: ${d.driver||"not detected"}\nAuto resolves to: ${d.auto_resolves_to}\n${d.core_error||d.note}\nWorkspace: ${ST.home}`}catch(e){$("#diag").textContent=e.message}}
document.addEventListener("click",e=>{if(e.target.matches(".figs img,.fig")){const l=document.createElement("div");l.className="lightbox";l.innerHTML=`<img src="${e.target.src}" alt="">`;l.onclick=()=>l.remove();document.body.appendChild(l)}});
$("#fs").onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
$("#quit").onclick=async()=>{if(!confirm("Close ProteusSegNet? Active jobs are never stopped."))return;try{await post("/api/shutdown",{});document.body.innerHTML='<p style="padding:40px;color:#8aa0b8">Proteus has stopped. You can close this tab.</p>'}catch(e){showErr(e.message)}};
setView("analysis");effective();refresh();
