// api/analyze.js - Vercel Serverless Function (CommonJS)

module.exports = async function handler(req, res) {
  // CORS 헤더 설정
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ status: 'error', message: 'Method Not Allowed' });
  }

  try {
    const { image, query, petProfile } = req.body || {};
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        status: 'error',
        message: 'GEMINI_API_KEY is not configured on Vercel environment variables.'
      });
    }

    const systemInstruction = `
당신은 대한민국 최고 수준의 반려동물 전문 수의학 AI 어드바이저입니다.
보호자가 전달한 반려동물(강아지/고양이)의 환부 사진(눈, 귀, 피부, 구강, 발 등)과 이상 증상 설명을 면밀히 분석하세요.

반드시 아래 규칙을 준수하세요:
1. 법적 의료법을 준수하여 "확정 진단"을 내리지 말고, "임상적 추정 및 의심 징후" 형태로 정중하고 신뢰감 있게 설명하세요.
2. 응답은 오직 순수한 JSON 형식만 반환해야 하며, 마크다운 코드블록(\`\`\`json 등)은 절대 포함하지 마세요.
3. JSON 규격은 반드시 아래 구조를 정확히 지키세요:
{
  "severity": "안정" | "주의" | "위험" (가장 적절한 위험도 1개),
  "observations": "사진 및 문진에서 관찰되는 객관적인 외관 특징 및 상태 요약 (2~3문장)",
  "candidates": [
    {
      "disease": "의심 질환명 1",
      "reason": "해당 질환을 의심하는 구체적인 임상적 이유와 증상 연관성"
    },
    {
      "disease": "의심 질환명 2",
      "reason": "해당 질환을 의심하는 구체적인 임상적 이유와 증상 연관성"
    }
  ],
  "checkpoints": [
    "보호자가 가정에서 면밀히 관찰해야 할 핵심 체크포인트 1",
    "보호자가 가정에서 면밀히 관찰해야 할 핵심 체크포인트 2",
    "동물병원 즉시 내원이 필요한 위험 신호 안내"
  ],
  "home_care": "동물병원 내원 전 또는 일상에서 보호자가 취할 수 있는 안전한 홈케어 및 응급 처치 요령 (2~3문장)"
}
`;

    const userParts = [];

    let promptText = `보호자 증상 질문: ${query || '사진의 환부 상태를 확인해 주세요.'}\n`;
    if (petProfile) {
      promptText += `반려동물 정보: 이름(${petProfile.name || '미등록'}), 종류(${petProfile.type || '강아지'}), 품종(${petProfile.breed || '모름'}), 나이(${petProfile.age || '모름'}), 체중(${petProfile.weight || '모름'})\n`;
    }
    userParts.push({ text: promptText });

    // Base64 이미지가 첨부된 경우
    if (image && typeof image === 'string') {
      let mimeType = 'image/jpeg';
      let base64Data = image;

      if (image.startsWith('data:')) {
        const matches = image.match(/^data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,(.+)$/);
        if (matches && matches.length === 3) {
          mimeType = matches[1];
          base64Data = matches[2];
        }
      }

      userParts.push({
        inline_data: {
          mime_type: mimeType,
          data: base64Data
        }
      });
    }

    const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=${apiKey}`;

    const geminiRes = await fetch(geminiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [
          {
            role: 'user',
            parts: userParts
          }
        ],
        system_instruction: {
          parts: [{ text: systemInstruction }]
        },
        generationConfig: {
          temperature: 0.2,
          responseMimeType: 'application/json'
        }
      })
    });

    if (!geminiRes.ok) {
      const errText = await geminiRes.text();
      console.error('[Gemini API Error]', errText);
      return res.status(geminiRes.status).json({
        status: 'error',
        message: 'Gemini API call failed',
        details: errText
      });
    }

    const geminiData = await geminiRes.json();
    const rawAiText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!rawAiText) {
      return res.status(500).json({
        status: 'error',
        message: 'No response content generated from Gemini'
      });
    }

    let parsedResult;
    try {
      const cleanJson = rawAiText.replace(/```json/g, '').replace(/```/g, '').trim();
      parsedResult = JSON.parse(cleanJson);
    } catch (parseErr) {
      console.error('[JSON Parse Error]', rawAiText);
      parsedResult = {
        severity: '주의',
        observations: rawAiText,
        candidates: [{ disease: '임상 관찰 필요', reason: '추가 검진 권장' }],
        checkpoints: ['24시간 내 증상 변화 관찰', '식욕 및 활력 점검'],
        home_care: '무리한 자극을 피하고 안정을 취하게 해주세요.'
      };
    }

    return res.status(200).json({
      status: 'success',
      source: 'gemini-3.8-flash',
      data: parsedResult
    });

  } catch (error) {
    console.error('[Serverless API Error]', error);
    return res.status(500).json({
      status: 'error',
      message: error.message || 'Internal Server Error'
    });
  }
};
