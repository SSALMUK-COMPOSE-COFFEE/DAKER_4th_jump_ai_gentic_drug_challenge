# 검증된 데이터 소스 (2026-07-30 실제 호출 확인)

전부 무료·인증 불필요(openFDA는 무키 시 rate limit 있음). 아래는 실제로 200 응답을 확인한 것만 기록.

## 대회 정보 조회 (DAKER)

공고 페이지는 SPA라 WebFetch로 본문이 안 읽힌다. 내부 API를 직접 쓸 것.

```bash
curl -sL "https://daker.ai/api/hackathons/4th-jump-ai-agentic-drug-challenge"        # 공고 본문·평가기준·일정·상금
curl -sL "https://daker.ai/api/hackathons/ed79c1f0-1a61-4615-aa13-74cd32956a59/posts" # 게시판 35건
```
`description` / `evaluationCriteria` / `rules` / `prizeDescription` 필드가 HTML이라 태그 제거 필요.

## 임상 / 규제

**ClinicalTrials.gov API v2** — 인증 없음, 페이지네이션 `pageToken`.
```bash
curl "https://clinicaltrials.gov/api/v2/studies?filter.overallStatus=TERMINATED&query.term=AREA%5BPhase%5DPHASE2&fields=NCTId,BriefTitle,WhyStopped,Phase,OverallStatus&pageSize=3&countTotal=true"
```
확인된 사실: **Phase 2 TERMINATED 시험이 9,794건**이고 대부분 `whyStopped` 자유서술이 채워져 있다 (`"Lack of efficacy"`, `"Recruitment difficulties"`, `"Development program terminated."` 등). 실패 라벨 코퍼스로 바로 쓸 수 있다.

**openFDA** — 약물 라벨, 이상반응(FAERS).
```bash
curl "https://api.fda.gov/drug/label.json?limit=1"
curl "https://api.fda.gov/drug/event.json?limit=1"
```

**FDA Guidance Documents** — API 아님, HTML 크롤링 필요. https://www.fda.gov/regulatory-information/search-fda-guidance-documents (200 확인)

**RxNav / RxNorm** — 약물명 정규화.
```bash
curl "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin"
```

## 타겟 / 질환

**Open Targets Platform GraphQL** — GET은 500 뜨므로 **POST 필수**.
```bash
curl -X POST https://api.platform.opentargets.org/api/v4/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"{target(ensemblId:\"ENSG00000146648\"){approvedSymbol associatedDiseases(page:{index:0,size:2}){count rows{disease{name} score}}}}"}'
```
EGFR → 연관 질환 6,459건, 각 쌍에 evidence score가 붙어 나온다. 타겟-질환 근거 랭킹에 그대로 활용 가능.

**UniProt REST**
```bash
curl "https://rest.uniprot.org/uniprotkb/P00533.json"
```

**AlphaFold DB**
```bash
curl "https://alphafold.ebi.ac.uk/api/prediction/P00533"
```

## 화합물 / 활성

**ChEMBL REST**
```bash
curl "https://www.ebi.ac.uk/chembl/api/data/molecule?limit=1&format=json"
```

**PubChem PUG REST**
```bash
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/CanonicalSMILES/JSON"
```

## 문헌

**PubMed E-utilities**
```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&retmax=1"
```

**Europe PMC** — 전문(full text) 접근이 PubMed보다 넓다. RAG용으로 이쪽이 유리.
```bash
curl "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=aspirin&format=json&pageSize=1"
```

## 로컬 라이브러리 (미설치, 설치 필요)

- **RDKit** — 심사표에 이름이 직접 등장. 분자 파싱·기술자·MMP·유효성 검증.
- **TDC (Therapeutics Data Commons)** — ADMET/docking oracle 벤치마크. 분야 2 갈 경우 정량 평가에 필수.
- **AiZynthFinder** — 역합성 경로 탐색.
