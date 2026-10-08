"""Real upload-cap encoding; run in its separately bounded CI lane."""
import wave
import numpy as np
import httpx
from tests.test_cloud_gap_regressions import mock_transport, network_guard

def test_real_upload_cap_split_and_overlap_only_speech(tmp_path,monkeypatch):
    import assessment_runtime.groq_asr as groq
    audio=tmp_path/'long-synthetic.wav'
    block=np.random.default_rng(13).integers(-32768,32767,16000,dtype=np.int16).astype('<i2').tobytes()
    with wave.open(str(audio),'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000)
        for _ in range(1200): output.writeframesraw(block)
    caps=[]
    def handler(request):
        body=request.content
        name=body.split(b'filename="')[1].split(b'"')[0].decode()
        start=body.index(b'fLaC');stop=body.index(b'\r\n--',start)
        caps.append(stop-start)
        assert 0 < caps[-1] <= 24_000_000
        # First half has speech only in its right overlap; second half owns it.
        offset=0 if name=='0a.flac' else 599
        return httpx.Response(200,json={'text':'neighbor','language':'english' if name=='0a.flac' else 'italian','words':[{'word':'neighbor','start':600.2-offset,'end':600.5-offset}]})
    mock_transport(monkeypatch,handler)
    result=groq.transcribe(audio,api_key='fixture',model='whisper-large-v3')
    assert [w['text'] for w in result['words']]==['neighbor']
    assert len(caps)==2
    assert result['detected_language']=='it'
