from common.prompt import PROMPT
from common.logger import logger

MAX_RETRIES = 5
INITIAL_DELAY = 1
MODEL = 'gemini-2.0-flash-001'

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
for attempt in range(MAX_RETRIES + 1):
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        logger.info("Gemini API 호출 완료")
        break
    except  errors.ServerError as e:
        logger.warning(f"Gemini API 서버 오류 (시도 {attempt + 1}/{MAX_RETRIES + 1}): {e}")
        if attempt < MAX_RETRIES:
            delay = INITIAL_DELAY * (2 ** attempt)
            logger.info(f"다음 시도까지 {delay}초 대기...")
            time.sleep(delay)
        else:
            logger.error("Gemini API 호출 실패 (최대 재시도 횟수 초과)")
            raise
    except errors.APIError as e:
        logger.error(f"Gemini API 클라이언트 오류: {e}")
        raise
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        raise
    else:
        pass

    if 'response' not in locals():
        logger.error("Gemini API 응답을 받지 못했습니다.")

    if 'response' in locals() and response:
        pass

result = json.loads(response.text)
