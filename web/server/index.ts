import express from 'express';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=fileURLToPath(new URL('../../',import.meta.url));
const python=process.env.PROJECT_PYTHON||path.join(root,'venv/bin/python');
const app=express();app.disable('x-powered-by');app.use(express.json({limit:'1kb'}));
function run(args:string[],timeout=30000):Promise<string>{return new Promise((resolve,reject)=>{
 const p=spawn(python,args,{cwd:root,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});let out='',err='';
 const timer=setTimeout(()=>{p.kill('SIGTERM');reject(new Error('The request timed out. The previous saved snapshot is still available.'));},timeout);
 p.stdout.on('data',d=>out+=d);p.stderr.on('data',d=>err+=d);p.on('error',e=>{clearTimeout(timer);reject(e)});p.on('close',code=>{clearTimeout(timer);code===0?resolve(out):reject(new Error(err.slice(-600)||'The project reader could not complete.'))});
});}
const reader=(kind:string,start?:string,end?:string)=>run([path.join(root,'web/server/read_project.py'),kind,...(start?[start,end!]:[])]).then(JSON.parse);
for(const name of ['live','pipeline','replay'])app.get('/api/'+name,async(req,res)=>{try{
 const start=String(req.query.start||'2024-09-01'),end=String(req.query.end||'2024-09-30');
 if(name==='replay'&&(!/^2024-09-\d{2}$/.test(start)||!/^2024-09-\d{2}$/.test(end)||start>end))return res.status(400).json({error:'Choose a valid September 2024 date range.'});
 res.set('Cache-Control','no-store').json(await reader(name,...(name==='replay'?[start,end]:[])));
}catch(e){res.status(503).json({error:(e as Error).message})}});
let fetching=false;
app.post('/api/live/fetch',async(req,res)=>{
 // Local browser only: prevent a cross-site page from triggering pollers.
 const origin=req.get('origin');if(origin&&!['http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8787','http://localhost:8787'].includes(origin))return res.status(403).json({error:'Local requests only.'});
 if(fetching)return res.status(409).json({error:'A refresh is already running. Please wait.'});fetching=true;
 try{const errors=JSON.parse(await run(['-c',"import json; from floodwatch.dashboard.live_monitor import refresh_sources; import contextlib,sys;\nwith contextlib.redirect_stdout(sys.stderr): errors=refresh_sources()\nprint(json.dumps(errors))"],240000));const data=await reader('live');res.status(errors.length?207:200).json({...data,refresh_errors:errors});}
 catch(e){res.status(502).json({error:(e as Error).message})}finally{fetching=false}
});
app.use(express.static(path.join(root,'web/client/dist')));app.get('/{*path}',(req,res)=>res.sendFile(path.join(root,'web/client/dist/index.html')));
const port=Number(process.env.PORT||8787);app.listen(port,'127.0.0.1',()=>console.log(`Web API: http://127.0.0.1:${port}`));
