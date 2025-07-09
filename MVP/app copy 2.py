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
    1.  위 정보를 바탕으로, 다음 개발 파트들의 공수(MM)를 예측해주세요
    2. 공수 예측에 대한 이유도 근거를 들어서 논리적으로 작성해야하고, 이를 'reason' 에 표시해야한다.
    3. 각 파트별 공수(MM)는 반드시 아래의 key 이름만 사용하여 정수 값으로 예측해주세요.  
- 요금제: KOS_Order_MM, KOS_Con_MM, APILink_MM, Bill_MM, APP_MM, MEIN_MM  
- 부가상품: Addon_KOS_Order_MM, Addon_KOS_Con_MM, Addon_APILink_MM, Addon_Bill_MM, Addon_APP_MM, Addon_MEIN_MM  
- 프로모션: Promo_KOS_Order_MM, Promo_KOS_Con_MM, Promo_APILink_MM, Promo_Bill_MM, Promo_APP_MM, Promo_MEIN_MM  
예측 결과에 해당 유형이 없으면 해당 key는 0으로 출력하세요.  
**절대로 다른 key 이름을 사용하지 마세요.**
    4. 예측 결과는 JSON 형식으로만 출력해주세요. 다른 설명은 일절 포함하지 마세요.
    5.  각 파트별 공수(MM)는 정수 값으로 예측해주세요.
    6.  JSON 형식은 다음과 같아야 합니다:
        ```json
        {{
            "KOS_Order_MM": <예측 MM>,
            "KOS_Con_MM": <예측 MM>,
            "APILink_MM": <예측 MM>,
            "Bill_MM": <예측 MM>,
            "APP_MM": <예측 MM>,
            "MEIN_MM": <예측 MM>,
            "Addon_KOS_Order_MM": <예측 MM>,
            "Addon_KOS_Con_MM": <예측 MM>,
            "Addon_APILink_MM": <예측 MM>,
            "Addon_Bill_MM": <예측 MM>,
            "Addon_APP_MM": <예측 MM>,
            "Addon_MEIN_MM": <예측 MM>,
            "Promo_KOS_Order_MM": <예측 MM>,
            "Promo_KOS_Con_MM": <예측 MM>,
            "Promo_APILink_MM": <예측 MM>,
            "Promo_Bill_MM": <예측 MM>,
            "Promo_APP_MM": <예측 MM>,
            "Promo_MEIN_MM": <예측 MM>,
            "reason": <예측 근거>
        }}
        ```
    """

    print("💬 LLM에게 공수 예측을 요청 중입니다...")
    # 4. LLM 호출
    try:
        INSTRUCTION = """\
너는 개발 공수 측정에 대한 전문가야. 개발 공수 관련해서 논리적인 사고를 하면서 답변을 줘야해
데이터 해석은 아래를 참고해줘

### 개발공수 데이터 해석
    KOS_Order_MM:유선오더통합팀에서 담당하고, 처음 요금제 관련 요구사항이 확정이되면, 청약을 위한 조건, 관련 UI개발, 기존 요금제와의 영향도 등에 따라 개발 필요하고 외부 API 연동이나 관련 시스템 들과의 연동을 위한 기본 데이터 생성등도 고려하는 개발 공수 
    KOS_Con_MM:무선고객개발팀에서 담당하고, 외부 API 연동이 있을때 관련 In/Out 데이터정의, 기존데이터와의 연동, 기존 요금제와의 영향도 등을 고려하는 개발공수
    APILink_MM: 결제플랫폼팀에서 담당하고, 외부  API  연동이 있을때 관련 In/Out 데이터정의, 기존데이터와의 연동 등을 고려하는 개발공수
    Bill_MM: 빌링개발팀에서 담당하고, 신규 요금제의 청구를 위한 요금항목 정의, 오더의 데이터를 기반하여 청구를 위한 데이터 생성, 할인이나 프로모션 등을 고려하여 최종 청구금액을 계산하는 개발공수
    APP_MM: 오픈서비스개발팀에서 담당하고, 기존 마이페이지라는 kt의 대고객 앱서비스에 새로 생기는 신규 요금제 추가에 따른 UI 개발, kt고객이 외부 OTT사의 최초 가입 엑티베이션을 위한 link 제공 및 관련 기능 구현, 테스트 등을 포함하는 개발공수
    MEIN_MM: 미디어채널개발팀에서 담당하고, 신규 요금제의 채널 권한을 제어하는 설계, 관련하여 KOS오더의 청약정보 연동 및 변경사항 연동 , 테스트 등을 포함하는 개발공수

### 개발공수 산정 방법:

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
        print("   예측 결과:\n", predicted_effort)
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
            part_name = part.replace("_MM", "")  # _MM 제거
            print(f"| {part_name:<10} | {effort:<14} |")
    print("---------------------------------")
    print(f"공수 예측 근거\n : {predicted_effort['reason']}")


# --- 프로그램 실행 ---
if __name__ == "__main__":
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="KT IPTV 요금제 개발 공수 예측기", layout="centered")

    st.title("KT IPTV 요금제 개발 공수 예측기")
    st.write("신규 프로젝트 요건을 입력하면, 과거 유사 프로젝트 데이터를 기반으로 개발 공수를 예측합니다.")

    project_req = st.text_area(
        "신규 프로젝트 요건을 입력하세요.",
        height=200,
        value="""새로운 OTT 서비스 'Wrtn TV'를 IPTV 요금제에 추가 연동하는 프로젝트입니다.
    주요 기능은 사용자 로그인 연동, 콘텐츠 탐색 (VOD 및 라이브 채널), 개인화 추천, 시청 이력 동기화입니다.
    기존 TVING 연동과 유사하지만, 새로운 OTT 플랫폼과의 연동이므로 초기 연동 작업이 더 필요할 수 있습니다.
    특히, Wrtn TV의 독점 콘텐츠를 강조하는 UI/UX 개선이 중요합니다."""
    )

    if st.button("공수 예측 실행"):
        with st.spinner("공수 예측 중..."):
            predicted = estimate_development_effort(project_req)
        if predicted:
            st.subheader("📊 예측된 개발 공수 (MM)")

            # 1. 예측 결과에서 파트명 동적 추출
            category_map = {
                "": "요금제",
                "Addon_": "부가상품",
                "Promo_": "프로모션"
            }
            # 예측 결과에서 모든 파트 추출 (reason 제외)
            all_parts = []
            for k in predicted.keys():
                if k == "reason":
                    continue
                # 접두사 제거
                for prefix in category_map.keys():
                    if k.startswith(prefix):
                        part = k[len(prefix):].replace("_MM", "")
                        if part not in all_parts:
                            all_parts.append(part)
                        break

            # 2. 데이터 분류
            table_dict = {part: {cat: "" for cat in category_map.values()} for part in all_parts}
            for k, v in predicted.items():
                if k == "reason":
                    continue
                for prefix, cat in category_map.items():
                    if k.startswith(prefix):
                        part = k[len(prefix):].replace("_MM", "")
                        if part in table_dict:
                            # 0이면 표시하지 않음
                            table_dict[part][cat] = f"{int(v)} MM" if int(v) != 0 else ""
                        break

            # 3. DataFrame 생성
            df = pd.DataFrame([
                {"개발 파트": part, **cats} for part, cats in table_dict.items()
            ])
            # 모든 열이 빈 값("")인 행(총공수 제외)은 제거
            df = df.loc[~((df.drop(columns=["개발 파트"]) == "").all(axis=1))]

            # 4. 총공수 계산
            total_row = {"개발 파트": "총공수"}
            for cat in category_map.values():
                s = [int(table_dict[part][cat].replace(" MM", "")) for part in all_parts if table_dict[part][cat]]
                total_row[cat] = f"{sum(s)} MM" if s else ""
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

            st.table(df)
            st.markdown(f"**공수 예측 근거:**\n\n{predicted['reason']}")
        else:
            st.error("공수를 예측할 수 없습니다. 다시 시도해주세요.")

