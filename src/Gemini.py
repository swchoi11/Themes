from prompt import PROMPT
from logger import logger

prompt = PROMPT.visibility_prompt()
image = client.files.upload(file=image_path)

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "required": [],
        "properties": {
            "": {"type": ""},
        }
    }
}

logger.info("Gemini API 호출 시작")
for attempt in range(self.args.max_retries + 1):
    try:
        response = self.client.models.generate_content(
            model=self.args.model,
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        self.logger.info("Gemini API 호출 완료")
        break
    except  errors.ServerError as e:
        self.logger.warning(f"Gemini API 서버 오류 (시도 {attempt + 1}/{self.args.max_retries + 1}): {e}")
        if attempt < self.args.max_retries:
            delay = self.args.initial_delay * (2 ** attempt)
            self.logger.info(f"다음 시도까지 {delay}초 대기...")
            time.sleep(delay)
        else:
            self.logger.error("Gemini API 호출 실패 (최대 재시도 횟수 초과)")
            raise
    except errors.APIError as e:
        self.logger.error(f"Gemini API 클라이언트 오류: {e}")
        raise
    except Exception as e:
        self.logger.error(f"예상치 못한 오류 발생: {e}")
        raise
    else:
        pass

    if 'response' not in locals():
        self.logger.error("Gemini API 응답을 받지 못했습니다.")
        # 적절한 에러 처리 또는 기본값 설정

    if 'response' in locals() and response:
        # response 처리 로직
        pass

result = json.loads(response.text)
