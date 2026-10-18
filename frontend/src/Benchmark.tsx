import React,{useEffect,useState} from 'react';
const pct=(value:number|null|undefined)=>value==null?'—':`${(value*100).toFixed(1)}%`;
const names:Record<string,string>={'Local semantic v4':'PlateBack','anthropic/claude-opus-5.5':'Claude Opus 5.5','anthropic/claude-sonnet-5.5':'Claude Sonnet 5.5','openai/gpt-6-luna':'GPT-6 Luna','openai/gpt-6.1-sol':'GPT-6.1 Sol','x-ai/grok-4.7':'Grok 4.7','qwen/qwen3.5-9b':'Qwen 3.5 · 9B'};
const label=(name:string)=>names[name]||name.replace(/^(anthropic|google|openai|qwen|x-ai)\//,'');
export default function Benchmark(){
 const[data,setData]=useState<any>(null);const[error,setError]=useState(false);const metric='mask_iou';
 useEffect(()=>{fetch('/api/benchmark').then(r=>{if(!r.ok)throw Error();return r.json()}).then(setData).catch(()=>setError(true))},[]);
 if(error)return <p role="alert">Could not load model scores. Refresh to try again.</p>;
 if(!data)return <p>Loading model scores…</p>;
 const display=(name:string)=>label(name);
 const models=[...(data.models||[])].filter((m:any)=>!m.model.startsWith('google/gemini')).filter((m:any)=>!m.model.startsWith('Local')||m.model==='Local semantic v4').sort((a:any,b:any)=>(b[metric]??-1)-(a[metric]??-1));
 return <div className="card model-comparison">
  <div className="card-head"><h2>Food segmentation</h2></div>
  <p className="muted">12 FoodSeg103 plates · four-food mean mask IoU. Higher is better. Public-data pilot; chicken masks include duck.</p>
  <section aria-label="Model mask IoU comparison" className="model-graph">
   {models.map((m:any)=><div className="benchmark-bar" key={m.model}><span>{display(m.model)}</span><div><i style={{width:`${Math.max(0,Math.min(100,(m[metric]??0)*100))}%`,background:m.model.startsWith('Local')?'#788c39':'#aaa99b'}}/></div><strong>{pct(m[metric])}</strong></div>)}
   <div className="model-axis"><span>0%</span><span>50%</span><span>100%</span></div>
  </section>
  <div className="table-scroll"><table><caption className="model-caption">Scores on the same benchmark images</caption><thead><tr><th>Model</th><th>Mask IoU</th></tr></thead><tbody>{models.map((m:any)=><tr key={m.model}><td>{display(m.model)}</td><td>{pct(m.mask_iou)}</td></tr>)}</tbody></table></div>
 </div>
}
