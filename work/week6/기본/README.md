# 📷 이미지 기반 QA RAG 봇
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![langchain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/sosososoyoen/streamlit-chatbot?quickstart=1)

## Overview
- [x]  여러 이미지를 입력으로 받기
- [x]  업로드 된 이미지들을 가지고 자유롭게 질의응답 할 수 있는 챗봇 구현
    - 채팅 내역을 prompt로 사용
    - 그리고 사용자가 여러 번 질문을 입력해도 처음 주어진 이미지들로 답변할 수 있도록 이미지를 벡터DB에 저장
- [x]  다음 이미지들과 질문에 대한 챗봇의 답변 생성
        - 이미지: 인터넷에서 강아지 사진과 고양이 사진 각각 1장씩 찾아 입력으로 쓰시면 됩니다.
        - 질문 1: 주어진 두 사진의 공통점이 뭐야?
        - 질문 2: 주어진 두 사진의 차이점이 뭐야?

[streamlit-app_img_basic-2025-05-09-02-05-19.webm](https://github.com/user-attachments/assets/b2d3cd04-47de-47ea-9682-3ed5f0e46131)


## Demo App
https://app-chatbot-gkqvxwqzjxqytmb8e8e9dp.streamlit.app/
