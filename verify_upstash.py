import os, json, uuid
try:
    import redis, rq as _rq
    url=os.environ['REDIS_URL']
    out={'endpoint': 'rediss'}
    r=redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=10)
    out['ping']=bool(r.ping())
    keyp=f"digits:test:{uuid.uuid4().hex[:8]}"
    set_key=keyp+':set'
    out['sadd']=int(r.sadd(set_key,'x'))
    out['sismember']=bool(r.sismember(set_key,'x'))
    z_key=keyp+':z'
    out['zadd']=int(r.zadd(z_key,{'a':1}))
    out['zrange']=r.zrange(z_key,0,-1)
    h_key=keyp+':h'
    out['hset']=int(r.hset(h_key, mapping={'field':'val'}))
    out['hget']=r.hget(h_key, 'field')
    out['publish']=int(r.publish('digits:events','watcher-upstash-verify'))
    # RQ registry availability
    has={'stopped': False, 'canceled': False}
    reg = getattr(_rq, 'registry', None)
    if reg is not None:
        has['stopped'] = bool(getattr(reg, 'StoppedJobRegistry', None))
        has['canceled'] = bool(getattr(reg, 'CanceledJobRegistry', None))
    out['rq_has']=has
    # instantiate registries if present
    inst={}
    conn=redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=10)
    q=_rq.Queue('digits', connection=conn)
    if has['stopped']:
        try:
            stopreg = reg.StoppedJobRegistry(queue=q)
            inst['stopped_ids']=stopreg.get_job_ids()
        except Exception as e:
            inst['stopped_error']=type(e).__name__
    if has['canceled']:
        try:
            canreg = reg.CanceledJobRegistry(queue=q)
            inst['canceled_ids']=canreg.get_job_ids()
        except Exception as e:
            inst['canceled_error']=type(e).__name__
    print(json.dumps({'ok': True, 'redis': out, 'rq': inst}, separators=(',',':')))
except Exception as e:
    print(json.dumps({'ok': False, 'error': f"{type(e).__name__}:{e}"}))
