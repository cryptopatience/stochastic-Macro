# 투자 리서치 대시보드

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Discord 웹훅

알림 URL은 코드에 넣지 않습니다. 아래 둘 중 하나만 설정하면 됩니다. 값이 없으면 전송만 건너뛰고 페이지는 그대로 열립니다.

설정 이름: `DISCORD_WEBHOOK_URL`

SSO 페이지(`pages/1_SSO.py`), `stochastic.py`, AI 종합분석(`pages/3_AI_종합분석.py`)은 같은 웹훅을 사용합니다.

우선순위는 환경변수가 먼저이고, 없으면 Streamlit secrets입니다.

로컬 또는 Streamlit Community Cloud:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml` (커밋하지 않음):

```toml
DISCORD_WEBHOOK_URL = ""
```

값에는 Discord 채널 설정의 웹훅 URL을 넣습니다. 셸 또는 배포 환경에서는 같은 이름을 환경변수로 줍니다.

```bash
export DISCORD_WEBHOOK_URL="웹훅 URL"
```

GitHub Actions 워크플로는 이 저장소에 없습니다. 나중에 추가할 때는 저장소 secret `DISCORD_WEBHOOK_URL`을 같은 이름의 환경변수로 넘기면 됩니다.

```yaml
env:
  DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
```
