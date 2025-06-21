from google import genai
from google.genai import types
import base64
import time


def generate(client, lines):
  csv_data = "\n".join(lines)
  msg1_text1 = types.Part.from_text(text=f"""CSV 에서 필요한 정보를 추출해서 Example 형식에 맞춰 출력해 주세요. 
- CSV는 \"sequence number, filename, description\" 으로 구성되어 있습니다.
- description은 테이블을 포함한 markdown 입니다. 여기에는 UI 컴포넌트 level, reason, score 정보가 포함되어 있습니다.

결과물은 코드가 아닌 CSV 입니다.
 * OUTPUT Format Example를 참조하여 csv 형태로 print 해주세요.
 * csv 내용만 출력하고, 그 외의 내용은 출력하지 마세요. 
 * 마크다운 (```csv)블럭은 사용하지 마세요.

### 출력 포맷
filename-01,UI 컴포넌트 label, reason, score
filename-01,UI 컴포넌트 label, reason, score

### CSV 데이터
{csv_data}
""")

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  output =""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    output += chunk.text
  return output

lines = []
with open("report.csv", "r", encoding='utf-8') as f:
  lines = f.readlines()

client = genai.Client(
    vertexai=True,
    project="gai-llm-poc",
    location="global",
)

batch_size = 5
count = 0

out_f = open("report_out.csv", "w", encoding="utf-8") 

for batch_start in range(0, len(lines), batch_size):
  batch = lines[batch_start:batch_start + batch_size]
  success = False
  for attempt in range(3):
    time.sleep(1)
    try:
      result = generate(client, batch)
      out_f.write(result)
      out_f.write("\n")
      success = True
      print(f">> 처리 {batch_start+5}/{len(lines)}")
      break
    except Exception as e:
      time.sleep(5)
  if not success:
    with open("error.txt", "w", encoding='utf-8') as f:
      f.writelines(lines)
