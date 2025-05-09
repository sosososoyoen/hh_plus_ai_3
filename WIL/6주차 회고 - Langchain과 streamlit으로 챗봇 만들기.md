
![Pasted image 20250509215906](https://github.com/user-attachments/assets/3aef442d-35fe-4034-bcc2-07ece531e53e)


이번 6주차는 5월 3일 휴강을 하게 되면서 2주로 연장되었다ㅎㅎ 오예~
연휴동안 쉬면서 프로젝트를 진행해보려고 했는데 하하 연휴 때 방송대 수업이나 여행으로 너무 바쁘게 움직여서 생각보다 많이 공부를 하지 못했다 흑흑...

6주차 발제 내용은 크게 3가지를 다뤘다.
1. Langchain과 Streamlit으로 챗봇 구현하기
2. LLM 챗봇 서빙하기
3. 멀티 모달 및 RAG를 LLM 챗봇에서 활용하기

거의 실습 코드 위주로 진행됐고, 이번 주부터 본격적으로 프로젝트 작업에 들어가는 거라서 발제 내용은 따로 정리하지 않을 것이다.

오픈소스로 풀린 LLM을 사용하기 위해 AWS 클라우드 환경을 미리 세팅해두었는데, 학습용 인스턴스 생성은 AWS 할당량 증가를 요청 해야 해서 하루 이틀 정도 걸린다.
다행히 발제 전에 세팅 튜토리얼이 올라오니까 인프라 모르는 사람들도 크게 걱정할 필요는 없을 것 같다.


## 🍀기본 과제 회고: 이미지를 가지고 질의응답을 하는 챗봇 구현

[데모 앱 보러가기](https://sosososoyoen-streamlit-chatbot-app-img-basic-7hq6m0.streamlit.app/)

[코드 보기](https://github.com/sosososoyoen/streamlit-chatbot/blob/main/app_img_basic.py)
지금 openAI API 2달러 밖에 안남아서 제대로 작동을 안할 수 있다.
추후에 자신의 API 키를 넣는 인풋을 넣어두겠다.

[streamlit-app_img_basic-2025-05-09-02-05-19.webm](https://github.com/user-attachments/assets/6a314550-32f3-457f-851c-9ffdbea17d11)


여러 이미지를 입력 받고 해당 이미지를 기반으로 QnA를 할 수 있는 챗봇을 구현했다.

**과제 요구 사항**
- [x]  여러 이미지를 입력으로 받기
- [x]  업로드 된 이미지들을 가지고 자유롭게 질의응답 할 수 있는 챗봇 구현
    - 채팅 내역을 prompt로 사용
    - 그리고 사용자가 여러 번 질문을 입력해도 처음 주어진 이미지들로 답변할 수 있도록 이미지를 벡터DB에 저장
- [x]  다음 이미지들과 질문에 대한 챗봇의 답변 생성 - 이미지: 인터넷에서 강아지 사진과 고양이 사진 각각 1장씩 찾아 입력으로 쓰시면 됩니다. 
	- 질문 1: 주어진 두 사진의 공통점이 뭐야? 
	- 질문 2: 주어진 두 사진의 차이점이 뭐야?


솔직히 이번 기본 과제에서 많이 헤맸다...
파이썬 기초 밖에 모르는 초보자기도 했고, Langchain와 RAG를 제대로 다뤄보는 건 처음이라 애를 많이 먹었다. 실제로 여태까지 한 과제들 중에서 가장 많은 시간을 들였다.
특히 이미지를 벡터 스토어에 저장해서 모든 유저의 입력에 이미지의 context를 불러오게 하는 부분에서 많이 헤맸다. 

1. RAG를 어떻게 관리해야하는가?
	1. 벡터 DB를 로컬에 저장해서 persistent하게 관리할 것인가?
	2. 아니면 인메모리로 관리해서 서버를 재시작할 때마다 벡터 DB를 초기화해야하나?

나는 처음에 로컬에 벡터 DB를 저장해두는 방식을 택했다.
그런데 여러 번 테스트를 해보니, 이전에 올려둔 이미지를 계속 context로 쌓여서 오히려 답변 생성에 방해가 됐다. 
``` python
def get_vectorstore() -> Chroma:
    client = chromadb.Client()
    # clear_system_cache() 이 부분은 streamlit 오류 관련으로 적은 코드입니다.
    client.clear_system_cache()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
```
그래서 예전에 썼던 예제들처럼 인메모리로 관리를 하게끔 했다.


2. 벡터 DB에 이미지를 어떻게 저장할 것인가?

처음에는 이미지를 임베딩 모델에 통과시켜야 하나 싶었는데,
내 목표는 GPT에 여러 장의 사진을 보내주는 거라 Base64로 인코딩한 이미지를 `page_content`에 저장하는 방식으로 작업했다.

```python
def images_to_docs(images: list) -> list[Document]:
    docs = []
    for image in images:
        img = Image.open(image)
        buffered = BytesIO()
        img.save(buffered, format=img.format)
        # base64로 인코딩
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Document 형태로 page_content 안에 
        docs.append(Document(
            page_content=f"data:image/{img.format.lower()};base64,{image_base64}",
        ))
    return docs
```

사용자의 입력을 쿼리로 사용해서 관련된 이미지들을 가져오고,
이 이미지들을 openAi API 문서에서 요구하는 형태의 구조로 가공했다.
https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded
```python
        retriever = vectordb.as_retriever()
        docs = retriever.get_relevant_documents(prompt)
        formatted_docs = format_docs(docs)
        #(중략)
        for image_data in formatted_docs:
            img_msgs[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })

        
```

### 아직 해결하지 못한 이슈들 

3. 파일 업로더에서 이미지 삭제하면 RAG에 쌓인 이미지 데이터도 함께 지워야 하는데, 어떻게 동기화할까?

떠오르는 방법은 업로드할 때 벡터 DB에 고유 ID를 붙여 두고, 렌더링할 때마다 현재 업로드된 ID 집합과 DB의 ID 집합을 비교해서 싱크를 맞추거나
업로드 리스트가 바뀔 때마다 벡터 스토어를 통째로 재생성하는 방식
근데 이게 Streamlit에서 깔끔하게 구현될지는 좀 더 확인해봐야 할 것 같다. 프로토타입 뚝딱 만들기엔 편한데 디테일한 기능을 구현하는 것에는 제약이 있다.

4. 유저가 올린 이미지들을 RAG로 관리하는 것이 맞을까? 

솔직히 이건 아직 확신이 서지 않는다. 
업로더에 올라온 이미지만 고려해서 답변해주면 되는 플로우라면, 굳이 RAG를 쓸 필요 없이 세션 메모리로도 충분히 관리할 수 있다.
그래서 심화 과제에서는 벡터 DB 대신 세션 메모리에 저장해서 처리했다.
물론 챗지피티처럼 채팅마다 이미지를 첨부할 수 있는 플로우일 경우에는 RAG가 유리하겠지만... 이 프로젝트에서는 좀 더 들여다볼 필요가 있을 것 같다. 

### 💥 streamlit cloud에 chromaDB 사용하는 app 배포할 때 주의해야할 점!!

streamlit 앱을 streamlit cloud에 배포할 때 chromadb를 설치하는 과정에서 의존성 에러가 발생한다.
오류 로그에 있는 문서 링크를 타고 들어가서 `pysqlite3-binary`를 설치한 후, 해당 소스코드 파일에 이 코드를 추가하면 된다.
```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3 as sqlite3
```


## 🌼심화 과제 회고: 이미지 기반 패션 추천 서비스

[데모 앱 보러가기](https://sosososoyoen-streamlit-chatbot-app-img-zikc9u.streamlit.app/)
따로 사이드바에서 본인의 openAI APi key를 입력해야 사용이 가능하다.

[코드 보기](https://github.com/sosososoyoen/streamlit-chatbot/blob/main/app_img.py)

[제목 없는 동영상 - Clipchamp로 제작 (1).webm](https://github.com/user-attachments/assets/d2d76cd1-092e-495c-b3d8-d878b9961260)


**과제 요구 사항 및 추가한 기능**
- [x]  여러 이미지를 입력으로 받아서 패션 추천 결과를 출력하기
- [x]  Langchain + Open AI(gpt-4o-mini)
- [x]  TavilySearchResults - 답변에서 키워드 추출해서 패션 아이템 검색

기본 과제에서는 RAG를 이미지 저장에 활용했다면
심화 과제에서는 웹 검색을 통해 답변의 퀄리티를 높이는 방향으로 작업을 했다.

원래는 논문 기반 QnA 서비스를 만들려고 했는데, 기본 과제에 힘을 너무 쏟아버린 나머지, 기본 과제와 비슷한 주제인 패션 추천 서비스로 주제를 바꿨다...
근데 솔직히 나는 패션 추천 주제에 관심이 매우 매우 없음... 또 멋대로 주제 바꿀지도 모른다

1. 웹 검색 도구를 어떻게 활용할까?
	
웹 검색 도구를 통한 RAG란 게 원래 답변을 생성할 때 context를 제공하는데 쓰이는 것인데 
나는 반대로 답변 생성 -> 키워드를 통해 검색 -> 참고할만한 문서들 가져와서 답변 밑에 링크로 달아주는 플로우로 작업했다.

```python
query_template = PromptTemplate.from_template("""
당신은 패션 스타일 전문 AI 어시스턴트입니다. 이미지 설명을 보고 그 사람에게 어울리는 스타일을 추천해주세요.
이미지 설명:
{caption}

유저 질문:
{user_input}

위 맥락을 참고해, 다음 형식으로 출력해줘:

답변: <전문적인 스타일 추천 텍스트>
키워드: <검색에 사용할 핵심 키워드 2~3개를 쉼표로 구분하여>
""")

# 사진을 분석하고 캡션을 만들어주는 함수
def generate_caption():
    if not st.session_state.image_data_list:
        return ""
    img_msgs = [
        {
            "role": "user", "content": [
                "위 이미지들을 설명하는 캡션을 한글로 만들어주세요."
            ]
        }
    ]
    for image_data in st.session_state.image_data_list:
        img_msgs[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_data['format']};base64,{image_data['base64']}"}
        })
    res = model.invoke(img_msgs)

    return res.content


# 답변 얻고 키워드 추출 후 웹 검색해서 답변 + 검색 결과 반환
def answer_and_search(caption: str, user_input: str, k: int = 3):
    # 2) 답변 + 키워드 생성
    raw = query_chain.invoke({"caption": caption, "user_input": user_input})
    
    # 3) 모델 출력 파싱
    answer = ""
    keywords = []
    for line in raw.split("\n"):
        if line.startswith("답변:"):
            answer = line[len("답변:"):].strip()
        if line.startswith("키워드:"):
            kws = line[len("키워드:"):].strip()
            keywords = [w.strip() for w in kws.split(",") if w.strip()]
    
    query = " ".join(keywords)
    search_results = search_docs(query, k)
    url_content_pairs = [{"url": doc.metadata["source"], "content": doc.page_content} for doc in search_results]

    return answer, keywords, url_content_pairs
```


```python
# 답변 만들어내는 코드
    with st.chat_message("assistant"):
        with st.spinner("이미지 분석 중...."):
            caption    = generate_caption()
        user_input = prompt
        
        if st.session_state.enable_search:
            with st.spinner("검색 중...."):
                answer, keywords, url_content_pairs = answer_and_search(caption, user_input, k=4)
                # 답변 먼저 보여주기
                st.markdown(answer)
                # 추출된 키워드
                st.caption("🔑 검색 키워드: " + ", ".join(keywords))
                # 이미지 출력
                for pair in url_content_pairs:
                    st.markdown(f"- [🔗 {pair['content']}]({pair['url']})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "keywords": keywords,
                    "urls": url_content_pairs
                })
        else:
            # 검색 꺼져있으면 그냥 답변만
            with st.spinner("답변 생성 중...."):
                qa_only = query_chain.invoke({"caption": caption, "user_input": user_input})
                st.markdown(qa_only)
                st.session_state.messages.append({"role": "assistant", "content": qa_only})
```

왜 이렇게 했냐면 웹 검색 도구 Tavily가 생각보다 검색을 못해서 그리고 시간이 부족해서 별 다른 아이디어가 나오지 않았다.

gpt한테 이미지들을 보내고 이미지들을 설명하는 캡션 만들어달라 하기 -> 캡션과 유저의 입력을 넣고 답변 생성 -> 생성된 답변에서 키워드를 추출해서 웹 검색 -> 답변과 함께 웹 검색 결과를 출력해주기

이 플로우로 작업을 해보았는데, gpt 토큰을 너무 많이 소비하기도 하고, 속도도 느려서 다른 방법을 찾아보려고 한다. 

지금 생각해보면 여러가지 의류 사진들을 임베딩 해서 벡터 스토어에 저장해두고, 유저가 추천해달라고 하면 비슷한 이미지 혹은 텍스트를 가져오는 방식이 괜찮아 보인다. 

### 🐋 앞으로 개선해나가야 하는 점

1. 유저의 이전 대화 내용을 기억해서 컨텍스트로 활용하기
	* https://python.langchain.com/docs/how_to/chatbots_memory/ 랭그래프를 활용해서 챗 히스토리 관리하기
	* 이미지 캡션 같은 부분을 매번 생성하지 않고, 캐싱하는 방식으로
2. Base64 인코딩으로 인한 페이로드 부피 최적화
	*  로컬에 경량 비전 모델을 두고 캡션, 임베딩을 처리해볼까..?
3. ✨ 제일 중요! 답변 내용의 평가 지표 정하고 평가하는 플로우 만들기
	*  제일 중요한 것이 이것인데... 기능 만들기에 급급해져서 여기에 대해 고민을 1도 하지 않았다. 
4. RAG를 어떻게 활용할 것인지 고민해보기
	1. RAG를 통해 답변의 퀄리티를 높이려면?
	2. 어떻게 하면 검색의 성능을 높일 수 있을까?


### 🐋 나의 목표가 무엇인지 인지하고 가자

나는 이 프로젝트에서 Langchain과 RAG를 최대한 잘 활용하여 gpt 4o까지는 아니어도 비교적 저비용의 모델을 사용해 최대한의 퀄리티를 끌어오는 것이다.
여기서 중요한건 RAG와 평가지표인데 남은 주차동안 여러가지를 시도해보아야겠다.

#항해99 #항해플러스 #AI #AI개발자  #항해플러스AI후기  #LLM #RAG #Langchain #streamlit
