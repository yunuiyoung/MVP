# 📦 IPTV 신규 상품 출시관련 개발 공수 구하기

## 🧭 개요
   사업부서에서 신규 상품 출시 원할시 적정한 개발 기간, 개발 예산을 ASAP으로 요구하시는데 관련 개발팀들의 노하우와 기존 경험에 비추어 공수를 각각 구하여 통합하여 제공합니다.
   유사한 개발 요청에는 기존의 경험기반으로 빠르고 정확한 답변을 할 수 있는 에이젼트 필요하여 개발 검토 하게됨.
   

## 🧭 아키텍쳐
① Azure AI Search로 과거 유사 사례를 찾고,   
② Azure OpenAI GPT 모델로 공수를 예측하며,   
③ Streamlit으로 사용자에게 결과를 시각적으로 보여주는 구조입니다.
   

## 🧭 주요기술
① Azure OpenAI Service   
   **LLM(대형 언어 모델)**을 통해 개발 공수 예측을 수행   
② Azure AI Search   
  유사 프로젝트를 찾기 위해 벡터 검색 기반 RAG에 사용   
  신규 프로젝트 설명 → 임베딩 → 벡터 유사도 기반 검색 수행   
③ Azure OpenAI Embedding 모델   
  사용자의 입력 텍스트를 벡터로 변환하여 AI Search와 연동   


## 🧭 시연   
user29-web-awdpexgxemdvhtb4.koreacentral-01.azurewebsites.net
  

