export async function bytes(url,expectedSize,sha){
  const cache=globalThis.caches?await caches.open('flybrain-models-v1').catch(()=>null):null;
  let response=cache?await cache.match(url):null;
  const verify=async r=>{
    const buffer=await r.arrayBuffer();
    if(expectedSize && buffer.byteLength!==expectedSize)throw new Error('Model size differs from the pinned release.');
    if(sha){
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
      if(hash!==sha)throw new Error('Model checksum failed. Nothing was loaded.');
    }
    return buffer;
  };
  if(response){try{return await verify(response);}catch{await cache.delete(url);}}
  response=await fetch(url,{signal:AbortSignal.timeout(45000)});
  if(!response.ok)throw new Error(`Model download failed (${response.status}). Retry when connected.`);
  const copy=response.clone(), buffer=await verify(response);
  if(cache)await cache.put(url,copy).catch(()=>{});
  return buffer;
}
