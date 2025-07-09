from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import json
import os
import pandas as pd # pandas 추가
from dotenv import load_dotenv

load_dotenv()

# --- 환경 변수 설정 (2단계에서 설정한 값과 동일하게) ---
# 이 부분은 실제 값을 채워 넣거나, 환경 변수로 설정해야 합니다.
service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY") # 실제 키로 변경 필요

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # 실제 엔드포인트로 변경 필요
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY") # 실제 키로 변경 필요
embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
llm_model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") # 배포된 LLM 모델 이름 (예: gpt-4, gpt-35-turbo)

# 환경 변수가 설정되지 않았다면 사용자에게 안내
if "YOUR_" in key or "YOUR_" in azure_openai_endpoint or "YOUR_" in azure_openai_key:
    print("🚨 경고: Azure 서비스 키 또는 엔드포인트가 설정되지 않았습니다.")
    print("   위 코드의 'YOUR_...' 부분을 실제 값으로 변경하거나, 환경 변수를 설정해주세요.")
    # exit() # 실제 운영 시에는 종료하거나 다른 방식으로 처리

# Azure OpenAI 클라이언트 초기화
openai_client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version="2024-12-01-preview"
)

# 임베딩 생성 함수
def generate_embeddings(text):
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_model_name
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ 임베딩 생성 중 오류 발생: {e}")
        print("   Azure OpenAI 엔드포인트, 키, 모델 배포 이름이 올바른지 확인해주세요.")
        return None

# Azure AI Search 클라이언트 초기화
search_client = SearchClient(endpoint=endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(key)
)

def estimate_development_effort(new_project_description: str):
    """
    신규 프로젝트 상세 요건을 바탕으로 개발 공수(MM)를 예측합니다.
    """
    print("✨ 새로운 프로젝트 요건을 분석 중입니다...")

    # # 1. 신규 요건 벡터화
    # new_description_vector = generate_embeddings(new_project_description)
    # if new_description_vector is None:
    #     return None
    # print("✅ 신규 요건 벡터화 완료.")

    # 2. Azure AI Search에서 유사 프로젝트 검색 (RAG)
    vector_query = VectorizableTextQuery(
        text=new_project_description,
        k_nearest_neighbors=3, # 최상위 3개 프로젝트 탐색
        fields="text_vector"
    )
    
    results = search_client.search(
        search_text=None, # 벡터 검색만 수행하므로 텍스트 검색은 비활성화
        vector_queries=[vector_query],
        # select=["project_name", "description", "dev_parts_effort"],
        top=3 # 상위 3개 결과만 가져오기
    )

    similar_projects = []
    for i, result in enumerate(results):
        similar_projects.append({
            "project_name": result["ProjectName"],
            "description": result["chunk"],
            'ProjectID': result["ProjectID"],
            'dev_parts_effort':{ 
                'APILink_MM': result["APILink_MM"], 
                'KOS_Order_MM': result["KOS_Order_MM"],
                'KOS_Con_MM': result['KOS_Con_MM'],
                'Bill_MM': result["Bill_MM"],
                'APP_MM': result["APP_MM"],
                'MEIN_MM': result["MEIN_MM"],
            }
        })
        print(f"🔍 유사 프로젝트 발견 ({i+1}위): {result['ProjectName']}")

    if not similar_projects:
        print("⚠️ 유사한 과거 프로젝트를 찾을 수 없습니다. 일반적인 추정치를 시도합니다.")
        # 유사 프로젝트가 없을 경우 기본 프롬프트로 대체
        context_str = "과거 프로젝트 데이터가 충분하지 않습니다. 일반적인 OTT 요금제 개발 파트별 공수 예측을 시도합니다."
    else:
        context_str = "다음은 유사한 과거 프로젝트들의 정보입니다:\n\n"
        for project in similar_projects:
            context_str += f"### 프로젝트 이름: {project['project_name']}\n"
            context_str += f"설명: {project['description']}\n"
            context_str += "과거 개발 공수 (MM):\n"
            for part, effort in project['dev_parts_effort'].items():
                context_str += f"  - {part}: {effort} MM\n"
            context_str += "\n"
    
    # 3. LLM 프롬프트 구성
    prompt = f"""
    당신은 KT IPTV 요금제 개발 공수를 예측하는 전문가입니다.
    새로운 프로젝트의 상세 요건과 유사한 과거 프로젝트 데이터를 참고하여, 각 개발 파트별 필요한 개발 공수(MM)를 매우 정확하게 산정해주세요.
    
    ---
    
    **새로운 프로젝트 상세 요건:**
    {new_project_description}
    
    ---
    
    **유사한 과거 프로젝트 정보:**
    {context_str}
    
    ---
    
    **요청 사항:**
    1.  위 정보를 바탕으로, 다음 개발 파트들의 공수(MM)를 예측해주세요: 프론트엔드, 백엔드, QA, 디자인, 기획.
    2. 공수 예측에 대한 이유도 근거를 들어서 논리적으로 작성해야하고, 이를 'reason' 에 표시해야한다.
    2.  각 파트별 공수(MM)는 정수 값으로 예측해주세요.
    3.  예측 결과는 JSON 형식으로만 출력해주세요. 다른 설명은 일절 포함하지 마세요.
    4.  JSON 형식은 다음과 같아야 합니다:
        ```json
        {{
            "KOS_Order_MM": <예측 MM>,
            "KOS_Con_MM": <예측 MM>,
            "APILink_MM": <예측 MM>,
            "Bill_MM": <예측 MM>,
            "APP_MM": <예측 MM>,
            "MEIN_MM": <예측 MM>,
            "reason": <예측 근거>
        }}
        ```
    """

    print("💬 LLM에게 공수 예측을 요청 중입니다...")
    # 4. LLM 호출
    try:
        INSTRUCTION = """\
너는 개발 공수 측정에 대한 전문가야. 개발 공수 관련해서 논리적인 사고를 하면서 답변을 줘야해
"""
        response = openai_client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # 창의성 조절 (0.0에 가까울수록 일관적, 1.0에 가까울수록 창의적)
            response_format={"type": "json_object"} # JSON 응답을 강제
        )
        llm_output = response.choices[0].message.content
        predicted_effort = json.loads(llm_output)
        print("🎉 공수 예측 완료!")
        return predicted_effort

    except json.JSONDecodeError as e:
        print(f"❌ LLM 응답 파싱 중 오류 발생: {e}")
        print("   LLM이 유효한 JSON을 반환하지 않았을 수 있습니다. LLM 응답:", llm_output)
        return None
    except Exception as e:
        print(f"❌ LLM 호출 중 오류 발생: {e}")
        print("   Azure OpenAI 엔드포인트, 키, 모델 배포 이름이 올바른지 확인해주세요.")
        return None


def display_effort_table(predicted_effort):
    """
    예측된 공수를 Markdown 테이블로 출력합니다.
    """
    if not predicted_effort:
        print("\n공수를 예측할 수 없습니다. 다시 시도해주세요.")
        return

    print("\n--- 📊 예측된 개발 공수 (MM) ---")
    print("| 개발 파트 | 예측 공수 (MM) |")
    print("|-----------|----------------|")
    for part, effort in predicted_effort.items():
        if part != "reason":
            print(f"| {part:<10} | {effort:<14} |")
    print("---------------------------------")
    print(f"공수 예측 근거\n : {predicted_effort['reason']}")


# --- 프로그램 실행 ---
if __name__ == "__main__":
    # 예시 신규 프로젝트 요건 입력
    new_project_req = """
    새로운 OTT 서비스 'Wrtn TV'를 IPTV 요금제에 추가 연동하는 프로젝트입니다.
    주요 기능은 사용자 로그인 연동, 콘텐츠 탐색 (VOD 및 라이브 채널), 개인화 추천, 시청 이력 동기화입니다.
    기존 TVING 연동과 유사하지만, 새로운 OTT 플랫폼과의 연동이므로 초기 연동 작업이 더 필요할 수 있습니다.
    특히, Wrtn TV의 독점 콘텐츠를 강조하는 UI/UX 개선이 중요합니다.
    """

    # 개발 공수 예측 실행
    estimated_mm = estimate_development_effort(new_project_req)

    # 결과 출력
    display_effort_table(estimated_mm)

