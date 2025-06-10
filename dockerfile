FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사 (레이어 캐싱 최적화)
COPY requirements.txt .

# Python 의존성 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일들 복사
COPY . .

# 출력 디렉토리 생성
RUN mkdir -p output output/jsons output/jsons/all_issues

# Python 경로 설정
ENV PYTHONPATH=/app

# 기본 명령어
CMD ["python", "main.py"]
