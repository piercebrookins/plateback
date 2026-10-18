import React,{useState} from 'react';
type Region={id:string;label:any;polygon:any[];holes:any[];score:number|null;food_name?:string|null;review_status?:'pending'|'confirmed'};
const name=(r:Region)=>(r.food_name||r.label).trim().toLowerCase();
const title=(s:string)=>s.charAt(0).toUpperCase()+s.slice(1);
export default function FoodReview({regions,selected,disabled,onSelect,onChange}:{regions:Region[];selected:string|null;disabled:boolean;onSelect:(id:string|null)=>void;onChange:(r:Region[])=>void}){
 const[editing,setEditing]=useState<string|null>(null),[draft,setDraft]=useState('');
 const groups=[...new Set(regions.map(name))];
 function rename(group:string){const value=draft.trim().toLowerCase();if(!value)return;const known=['rice','pasta','broccoli','chicken'].includes(value);onChange(regions.map(r=>name(r)===group?{...r,label:known?value:'unknown',food_name:known?null:value,review_status:'confirmed'}:r));setEditing(null)}
 return <div className="food-review-list">{groups.map(group=>{const members=regions.filter(r=>name(r)===group),confirmed=members.every(r=>r.review_status==='confirmed');return <section className={'food-review-item '+(members.some(r=>r.id===selected)?'selected':'')} key={group}>
 <button className="food-review-name" disabled={disabled} onClick={()=>onSelect(members[0].id)}><strong>{title(group)}</strong><small>{confirmed?'Confirmed':'Needs review'}</small></button>
 {editing===group?<form onSubmit={e=>{e.preventDefault();rename(group)}}><input autoFocus aria-label="Food name" maxLength={60} value={draft} onChange={e=>setDraft(e.target.value)} disabled={disabled}/><button disabled={disabled||!draft.trim()}>Save name</button><button type="button" onClick={()=>setEditing(null)}>Cancel</button></form>:<div className="food-review-actions">{!confirmed&&<button disabled={disabled} onClick={()=>onChange(regions.map(r=>name(r)===group?{...r,review_status:'confirmed'}:r))}>Confirm</button>}<button disabled={disabled} onClick={()=>{setEditing(group);setDraft(group)}}>Rename</button><button disabled={disabled} onClick={()=>{onChange(regions.filter(r=>name(r)!==group));if(members.some(r=>r.id===selected))onSelect(null)}}>Remove</button></div>}
 </section>})}</div>
}
