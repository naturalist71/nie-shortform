import os
import json
import asyncio
import textwrap
import glob
import random
import numpy as np
import streamlit as st
from pypdf import PdfReader
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. 환경 설정 및 디렉토리 생성
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="국립생태원 연구성과 숏폼 제작 & 웹 연계 시스템",
    page_icon="🌿",
    layout="wide"
)

IMAGE_DIR = "images"
AUDIO_DIR = "audios"
COMPOSITE_DIR = "composite_frames"
OUTPUT_DIR = "outputs"

for folder in [IMAGE_DIR, AUDIO_DIR, COMPOSITE_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. 5대 기후대(열대·사막·지중해·온대·극지) 실사풍 배경 생성
# ------------------------------------------------------------------------------
def create_5_biomes_backgrounds():
    width, height = 1080, 1920

    # 1. Scene 1: 열대관 (Tropical Rainforest)
    p1 = os.path.join(IMAGE_DIR, "scene_1_Hook.png")
    arr1 = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        r = y / height
        arr1[y, :, 0] = 15 * (1 - r) + 10 * r
        arr1[y, :, 1] = 45 * (1 - r) + 35 * r
        arr1[y, :, 2] = 30 * (1 - r) + 20 * r
    img1 = Image.fromarray(np.uint8(arr1)).convert("RGBA")
    ov1 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(ov1)
    for i in range(7):
        sx = 50 + i * 160
        d1.polygon([(sx, 0), (sx + 100, 0), (sx + 280, height), (sx + 120, height)], fill=(180, 240, 190, 22))
    for x in range(-100, width + 200, 120):
        d1.ellipse([x, 500, x + 350, 1200], fill=(20, 65, 45, 160))
        d1.ellipse([x - 50, 900, x + 280, 1600], fill=(12, 50, 30, 210))
        d1.polygon([(x, 1500), (x + 140, 1350), (x + 80, 1800)], fill=(8, 38, 22, 240))
    Image.alpha_composite(img1, ov1).filter(ImageFilter.GaussianBlur(1.0)).convert("RGB").save(p1)

    # 2. Scene 2: 사막관 (Desert Dunes)
    p2 = os.path.join(IMAGE_DIR, "scene_2_Problem.png")
    arr2 = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        r = y / height
        arr2[y, :, 0] = 225 * (1 - r) + 160 * r
        arr2[y, :, 1] = 135 * (1 - r) + 85 * r
        arr2[y, :, 2] = 70 * (1 - r) + 35 * r
    img2 = Image.fromarray(np.uint8(arr2)).convert("RGBA")
    ov2 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(ov2)
    for rad in range(350, 0, -25):
        d2.ellipse([540 - rad, 380 - rad, 540 + rad, 380 + rad], fill=(255, 235, 170, int(45 * (1 - rad / 350))))
    d2.polygon([(0, 850), (380, 720), (820, 840), (1080, 760), (1080, 1920), (0, 1920)], fill=(195, 115, 55, 230))
    d2.polygon([(0, 1100), (520, 930), (1080, 1080), (1080, 1920), (0, 1920)], fill=(165, 90, 40, 245))
    d2.polygon([(0, 1400), (420, 1260), (890, 1380), (1080, 1300), (1080, 1920), (0, 1920)], fill=(135, 70, 30, 255))
    Image.alpha_composite(img2, ov2).filter(ImageFilter.GaussianBlur(1.0)).convert("RGB").save(p2)

    # 3. Scene 3: 지중해관 (Mediterranean Coast)
    p3 = os.path.join(IMAGE_DIR, "scene_3_Research.png")
    arr3 = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        r = y / height
        if r < 0.45:
            nr = r / 0.45
            arr3[y, :, 0] = 110 * (1 - nr) + 195 * nr
            arr3[y, :, 1] = 175 * (1 - nr) + 225 * nr
            arr3[y, :, 2] = 235 * (1 - nr) + 250 * nr
        else:
            nr = (r - 0.45) / 0.55
            arr3[y, :, 0] = 20 * (1 - nr) + 10 * nr
            arr3[y, :, 1] = 85 * (1 - nr) + 55 * nr
            arr3[y, :, 2] = 145 * (1 - nr) + 95 * nr
    img3 = Image.fromarray(np.uint8(arr3)).convert("RGBA")
    ov3 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d3 = ImageDraw.Draw(ov3)
    d3.polygon([(0, 780), (320, 680), (640, 760), (850, 720), (1080, 790), (1080, 920), (0, 920)], fill=(185, 175, 145, 220))
    d3.polygon([(0, 850), (480, 980), (380, 1550), (0, 1700)], fill=(95, 125, 75, 240))
    for y in range(950, 1600, 120):
        d3.ellipse([40, y, 220, y + 100], fill=(65, 88, 50, 245))
    Image.alpha_composite(img3, ov3).filter(ImageFilter.GaussianBlur(1.0)).convert("RGB").save(p3)

    # 4. Scene 4: 온대관 (Temperate Forest)
    p4 = os.path.join(IMAGE_DIR, "scene_4_Finding.png")
    arr4 = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        r = y / height
        arr4[y, :, 0] = 25 * (1 - r) + 18 * r
        arr4[y, :, 1] = 68 * (1 - r) + 48 * r
        arr4[y, :, 2] = 42 * (1 - r) + 28 * r
    img4 = Image.fromarray(np.uint8(arr4)).convert("RGBA")
    ov4 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d4 = ImageDraw.Draw(ov4)
    for i in range(5):
        sx = 100 + i * 220
        d4.polygon([(sx, 0), (sx + 130, 0), (sx + 320, height), (sx + 140, height)], fill=(245, 255, 215, 30))
    trunks = [(110, 55), (280, 65), (510, 80), (740, 60), (920, 75)]
    for tx, tw in trunks:
        d4.rectangle([tx, 0, tx + tw, height], fill=(42, 32, 24, 250))
    Image.alpha_composite(img4, ov4).filter(ImageFilter.GaussianBlur(1.0)).convert("RGB").save(p4)

    # 5. Scene 5: 극지관 (Polar Glacier & Aurora)
    p5 = os.path.join(IMAGE_DIR, "scene_5_Impact.png")
    arr5 = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        r = y / height
        arr5[y, :, 0] = 8 * (1 - r) + 25 * r
        arr5[y, :, 1] = 22 * (1 - r) + 75 * r
        arr5[y, :, 2] = 55 * (1 - r) + 115 * r
    img5 = Image.fromarray(np.uint8(arr5)).convert("RGBA")
    ov5 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d5 = ImageDraw.Draw(ov5)
    for i in range(4):
        ay = 280 + i * 90
        d5.polygon([(0, ay), (380, ay - 80), (750, ay + 60), (1080, ay - 40), (1080, ay + 120), (0, ay + 150)], fill=(75, 240, 185, 45))
    d5.polygon([(0, 880), (280, 720), (520, 840), (780, 690), (1080, 810), (1080, 1920), (0, 1920)], fill=(160, 215, 235, 235))
    d5.polygon([(0, 1350), (580, 1260), (1080, 1330), (1080, 1920), (0, 1920)], fill=(190, 235, 248, 255))
    Image.alpha_composite(img5, ov5).filter(ImageFilter.GaussianBlur(1.0)).convert("RGB").save(p5)

create_5_biomes_backgrounds()

# ------------------------------------------------------------------------------
# 3. 폰트 로더 및 캐릭터 외곽선 정밀 투명화(누끼) 함수
# ------------------------------------------------------------------------------
def get_korean_font(size: int, bold: bool = False):
    font_candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf" if bold else "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/batang.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def apply_soft_feathered_border(char_img: Image.Image) -> Image.Image:
    char_img = char_img.convert("RGBA")
    w, h = char_img.size
    
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    margin = int(min(w, h) * 0.04)
    radius = int(min(w, h) * 0.16)
    mdraw.rounded_rectangle([margin, margin, w - margin, h - margin], radius=radius, fill=255)
    
    feather_mask = mask.filter(ImageFilter.GaussianBlur(12))
    orig_alpha = char_img.split()[3]
    combined_alpha = Image.fromarray(
        np.minimum(np.array(orig_alpha), np.array(feather_mask))
    )
    char_img.putalpha(combined_alpha)
    return char_img

def get_gwiyomi_and_pengi_mascots():
    search_files = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("images/*.png") + glob.glob("images/*.jpg")
    
    gwiyomi_files = []
    pengi_files = []
    other_mascots = []

    for f in set(search_files):
        f_name = os.path.basename(f).lower()
        if any(f_name.startswith(p) for p in ["scene_", "frame_", "test_", "thumbnail"]):
            continue

        if any(k in f_name for k in ["귀요미", "gwiyomi", "fox", "gwi"]):
            gwiyomi_files.append(f)
        elif any(k in f_name for k in ["팽이", "펭이", "pengi", "penguin", "peng"]):
            pengi_files.append(f)
        else:
            other_mascots.append(f)

    final_list = []
    if gwiyomi_files:
        final_list.append(gwiyomi_files[0])
    if pengi_files:
        final_list.append(pengi_files[0])
        
    if not final_list and other_mascots:
        final_list = other_mascots[:2]
        
    return final_list

def extract_pdf_text(uploaded_file, max_pages=8):
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    target_pages = min(len(reader.pages), max_pages)
    text_chunks = [reader.pages[i].extract_text() for i in range(target_pages) if reader.pages[i].extract_text()]
    return "\n\n".join(text_chunks)

def clean_and_parse_json(raw_text):
    raw = str(raw_text).strip()
    s_idx = raw.find("{")
    e_idx = raw.rfind("}")
    if s_idx != -1 and e_idx != -1:
        json_slice = raw[s_idx:e_idx + 1]
        return json.loads(json_slice)
    return json.loads(raw)

def generate_script_with_gemini(research_text, api_key):
    """60초 대본 및 Google Vids/Runway용 시네마틱 프롬프트 동시 생성"""
    client = genai.Client(api_key=api_key)
    prompt_lines = [
        "당신은 국립생태원의 수석 과학 커뮤니케이터이자 AI 시네마틱 영상 총괄 디렉터입니다.",
        "다음 연구보고서 텍스트를 바탕으로, 5개 씬 총 60초(1분) 교육용 숏폼 기획안을 작성하세요.",
        "동시에 각 씬별로 Google Vids, Runway Gen-3, Luma Dream Machine에서 사용할 수 있는 초고화질 시네마틱 영문 프롬프트(visual_prompt_en)를 함께 작성하세요.",
        "각 씬은 5대 기후대(1.열대 -> 2.사막 -> 3.지중해 -> 4.온대 -> 5.극지) 배경 순서에 어울리도록 구성하세요.",
        "",
        "[연구 텍스트]",
        str(research_text),
        "",
        "[작성 지침 - 5개 씬 총 60초 분량 & 시네마틱 비디오 프롬프트]",
        "- narration: 각 씬당 50~65자 내외의 친절하고 알찬 여성 구어체 나레이션 (~해요, ~랍니다 체).",
        "- caption: 12자 이내의 임팩트 있는 대형 핵심 키워드 자막",
        "- visual_prompt_en: Google Vids / Runway용 고화질 영문 프롬프트 (카메라 앵글, 렌즈 스펙, 조명, 8K 내셔널지오그래픽 다큐멘터리 스타일 포함).",
        "- 결과는 순수 JSON 형식으로만 반환하세요.",
        "",
        "[출력 형식 - JSON Only]",
        "{",
        '  "title": "연구 제목",',
        '  "research_summary": "연구 요약 2줄",',
        '  "scenes": [',
        '    { "scene_number": 1, "scene_type": "Hook", "narration": "...", "caption": "...", "visual_prompt_en": "Cinematic 8k footage of tropical rainforest mist, slow drone push-in, volumetric morning sunbeams, National Geographic documentary style" },',
        '    { "scene_number": 2, "scene_type": "Problem", "narration": "...", "caption": "...", "visual_prompt_en": "..." },',
        '    { "scene_number": 3, "scene_type": "Research", "narration": "...", "caption": "...", "visual_prompt_en": "..." },',
        '    { "scene_number": 4, "scene_type": "Finding", "narration": "...", "caption": "...", "visual_prompt_en": "..." },',
        '    { "scene_number": 5, "scene_type": "Impact", "narration": "...", "caption": "...", "visual_prompt_en": "..." }',
        "  ]",
        "}"
    ]
    prompt = "\n".join(prompt_lines)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    )
    
    return clean_and_parse_json(response.text)

def generate_female_voice(text, output_path):
    async def _tts():
        communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural", rate="+6%")
        await communicate.save(output_path)
    asyncio.run(_tts())

def make_composite_frame(bg_path, caption, narration, scene_num, out_path, mascot_file=None):
    base_img = Image.open(bg_path).convert("RGBA")
    width, height = base_img.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 1. 상단 배지 (42pt)
    badge_font = get_korean_font(42, bold=True)
    badge_text = f"국립생태원 연구성과 | Scene {scene_num}"
    bb = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    
    pad_x, pad_y = 38, 20
    b_x1, b_y1 = 60, 75
    b_x2, b_y2 = b_x1 + bw + (pad_x * 2), b_y1 + bh + (pad_y * 2)
    
    draw.rounded_rectangle([b_x1, b_y1, b_x2, b_y2], radius=24, fill=(15, 45, 25, 235), outline=(100, 220, 140, 255), width=3)
    draw.text((b_x1 + pad_x, b_y1 + pad_y - 4), badge_text, font=badge_font, fill=(255, 255, 255, 255))

    # 2. [중간 상단] 독립 핵심 자막 카드 (70pt)
    cap_font = get_korean_font(70, bold=True)
    c_bb = draw.textbbox((0, 0), caption, font=cap_font)
    cw, ch = c_bb[2] - c_bb[0], c_bb[3] - c_bb[1]

    cap_pad_x = 50
    cap_pad_y = 26
    cap_box_w = cw + (cap_pad_x * 2)
    cap_box_h = ch + (cap_pad_y * 2)
    
    cap_x1 = (width - cap_box_w) // 2
    cap_y1 = 590
    cap_x2 = cap_x1 + cap_box_w
    cap_y2 = cap_y1 + cap_box_h

    draw.rounded_rectangle([cap_x1, cap_y1, cap_x2, cap_y2], radius=30, fill=(10, 25, 18, 240), outline=(90, 240, 140, 255), width=5)
    draw.text((cap_x1 + pad_x, cap_y1 + pad_y - 4), caption, font=cap_font, fill=(255, 238, 60, 255))

    # 3. [하단] 세부 나레이션 박스 (38pt)
    narr_font = get_korean_font(38, bold=False)
    wrapped_narr = textwrap.fill(narration, width=21)
    n_bb = draw.textbbox((0, 0), wrapped_narr, font=narr_font)
    nw, nh = n_bb[2] - n_bb[0], n_bb[3] - n_bb[1]

    narr_box_w = min(width - 90, max(nw + 100, 800))
    narr_box_h = nh + 75
    
    narr_x1 = (width - narr_box_w) // 2
    narr_y1 = height - narr_box_h - 90
    narr_x2 = narr_x1 + narr_box_w
    narr_y2 = narr_y1 + narr_box_h

    draw.rounded_rectangle([narr_x1, narr_y1, narr_x2, narr_y2], radius=26, fill=(0, 0, 0, 220), outline=(255, 255, 255, 150), width=3)
    draw.text(((width - nw) // 2, narr_y1 + 36), wrapped_narr, font=narr_font, fill=(255, 255, 255, 255), align="center")

    # 4. [중간 자막과 아래 나레이션 사이] 부드러운 페더링 외곽선의 캐릭터 원본 배치
    if mascot_file and os.path.exists(mascot_file):
        try:
            raw_char = Image.open(mascot_file)
            soft_char = apply_soft_feathered_border(raw_char)
            
            target_h = 410
            c_ratio = target_h / float(soft_char.size[1])
            target_w = int(float(soft_char.size[0]) * c_ratio)
            char_resized = soft_char.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            shadow_mask = char_resized.split()[3].point(lambda p: 130 if p > 20 else 0)
            shadow_img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            shadow_img.paste((0, 0, 0, 110), (0, 0), shadow_mask)
            shadow_blurred = shadow_img.filter(ImageFilter.GaussianBlur(10))

            mid_gap_y = (cap_y2 + narr_y1) // 2
            char_pos_x = (width - target_w) // 2
            char_pos_y = mid_gap_y - (target_h // 2)

            overlay.paste(shadow_blurred, (char_pos_x + 4, char_pos_y + 6), shadow_blurred)
            overlay.paste(char_resized, (char_pos_x, char_pos_y), char_resized)
        except Exception:
            pass

    final = Image.alpha_composite(base_img, overlay).convert("RGB")
    final.save(out_path, quality=98)

def render_moviepy_video(plan_data, output_mp4_path="outputs/final_shortform.mp4"):
    create_5_biomes_backgrounds()
    clips = []
    scenes = plan_data.get("scenes", [])
    
    mascots = get_gwiyomi_and_pengi_mascots()
    chosen_single_mascot = random.choice(mascots) if mascots else None
    
    audio_durations = []
    audio_clips = []
    for scene in scenes:
        s_num = scene.get("scene_number")
        narration = scene.get("narration")
        audio_file = os.path.join(AUDIO_DIR, f"scene_{s_num}.mp3")
        generate_female_voice(narration, audio_file)
        
        aclip = AudioFileClip(audio_file)
        audio_clips.append(aclip)
        audio_durations.append(max(aclip.duration, 8.0))

    total_raw_time = sum(audio_durations)
    TARGET_TOTAL_TIME = 60.0
    scene_durations = [(d / total_raw_time) * TARGET_TOTAL_TIME for d in audio_durations]

    for idx, scene in enumerate(scenes):
        s_num = scene.get("scene_number")
        s_type = scene.get("scene_type")
        narration = scene.get("narration")
        caption = scene.get("caption")

        bg_file = os.path.join(IMAGE_DIR, f"scene_{s_num}_{s_type}.png")
        if not os.path.exists(bg_file):
            bg_file = os.path.join(IMAGE_DIR, f"scene_{s_num}_Hook.png")
            
        frame_file = os.path.join(COMPOSITE_DIR, f"frame_{s_num}.png")
        make_composite_frame(bg_file, caption, narration, s_num, frame_file, chosen_single_mascot)

        s_dur = scene_durations[idx]
        img_clip = ImageClip(frame_file)
        if hasattr(img_clip, "with_duration"):
            img_clip = img_clip.with_duration(s_dur).with_audio(audio_clips[idx])
        else:
            img_clip = img_clip.set_duration(s_dur).set_audio(audio_clips[idx])
            
        clips.append(img_clip)

    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(
        output_mp4_path,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )
    return output_mp4_path, chosen_single_mascot

# ------------------------------------------------------------------------------
# 4. Streamlit 웹 인터페이스 (3-Tab 독립 분리 구조)
# ------------------------------------------------------------------------------
st.title("🌿 국립생태원 연구성과 숏폼 제작 & 웹 연계 시스템")
st.markdown("내용 길이에 맞춰 유연하게 배분된 **공백 없는 총 60초(01:00)** 숏폼 비디오 제작 시스템입니다.")

st.sidebar.header("⚙️ API 설정")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

detected_mascots = get_gwiyomi_and_pengi_mascots()
if detected_mascots:
    st.sidebar.success(f"🐾 사용 가능 캐릭터: {', '.join([os.path.basename(m) for m in detected_mascots])}")
else:
    st.sidebar.info("💡 `eco_project` 폴더에 귀요미 또는 펭이 이미지(PNG)를 넣으시면 영상에 자동 합성됩니다.")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.plan_data = {
        "title": "",
        "research_summary": "",
        "scenes": [
            {
                "scene_number": i + 1,
                "caption": "",
                "narration": "",
                "visual_prompt_en": ""
            } for i in range(5)
        ]
    }
    for i in range(5):
        st.session_state[f"cap_{i}"] = ""
        st.session_state[f"narr_{i}"] = ""

# 3개의 독립 탭 구성
tab1, tab2, tab3 = st.tabs([
    "🎬 [1~4단계] 동영상 제작 & 렌더링", 
    "🌐 [5단계] 국립생태원 웹페이지 연계 코드", 
    "✨ [확장 옵션] Google Vids / 생성형 AI 시네마틱 프롬프트"
])

scene_headers = [
    "Scene 1 (Hook)", 
    "Scene 2 (Problem)", 
    "Scene 3 (Research)", 
    "Scene 4 (Finding)", 
    "Scene 5 (Impact)"
]

scene_types = ["Hook", "Problem", "Research", "Finding", "Impact"]

with tab1:
    uploaded_pdf = st.file_uploader("📄 연구보고서 또는 논문 PDF 업로드", type=["pdf"])

    if uploaded_pdf:
        if not api_key_input:
            st.warning("⚠️ 왼쪽 사이드바에 **Gemini API Key**를 입력해 주시면 PDF를 즉시 자동 분석합니다.")
        else:
            if "last_loaded_file" not in st.session_state or st.session_state.last_loaded_file != uploaded_pdf.name:
                with st.spinner(f"'{uploaded_pdf.name}' 분석 중... 5개 씬 총 60초 대본 및 AI 영상 프롬프트를 자동 생성합니다."):
                    try:
                        pdf_text = extract_pdf_text(uploaded_pdf)
                        new_plan = generate_script_with_gemini(pdf_text, api_key_input)
                        
                        st.session_state.plan_data = new_plan
                        st.session_state.last_loaded_file = uploaded_pdf.name
                        
                        for idx, sc in enumerate(new_plan.get("scenes", [])):
                            st.session_state[f"cap_{idx}"] = sc.get("caption", "")
                            st.session_state[f"narr_{idx}"] = sc.get("narration", "")
                        
                        with open("video_plan_step1.json", "w", encoding="utf-8") as f:
                            json.dump(new_plan, f, ensure_ascii=False, indent=2)
                        
                        st.success(f"✅ '{uploaded_pdf.name}' 분석 완료! 씬별 자막과 대본이 채워졌습니다.")
                    except Exception as e:
                        st.error(f"❌ PDF 분석 중 오류가 발생했습니다: {e}")

    st.markdown("### 📝 씬별 자막 및 나레이션 (내용별 유연 배분 · 총 60초, 직접 수정 가능)")
    cols = st.columns(5)
    
    current_scenes = []
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"**{scene_headers[idx]}**")
            c_val = st.text_input(
                f"💬 [중앙] 핵심 자막 ({idx+1})", 
                key=f"cap_{idx}",
                placeholder="화면 중간 70pt 대형 자막"
            )
            n_val = st.text_area(
                f"🗣️ [하단] 세부 나레이션 ({idx+1})", 
                key=f"narr_{idx}", 
                height=150,
                placeholder="화면 맨 아래 나레이션 (50~65자)"
            )
            
            old_scenes = st.session_state.plan_data.get("scenes", [])
            v_en = old_scenes[idx].get("visual_prompt_en", "") if idx < len(old_scenes) else ""
            
            current_scenes.append({
                "scene_number": idx + 1,
                "scene_type": scene_types[idx],
                "caption": c_val,
                "narration": n_val,
                "visual_prompt_en": v_en
            })
            
    st.session_state.plan_data["scenes"] = current_scenes

    st.markdown("---")
    
    if st.button("🎬 최종 고화질 MP4 동영상 렌더링 시작 (공백 없는 총 60초 완벽 맞춤)", use_container_width=True):
        has_empty = any(not s.get('narration') or not s.get('caption') for s in current_scenes)
        if has_empty:
            st.warning("⚠️ 자막과 나레이션을 입력한 후 렌더링을 진행해 주세요.")
        else:
            with st.spinner("소프트 페더링 캐릭터 합성 및 내용 맞춤형 60초 비디오를 렌더링 중입니다... (약 35~45초 소요)"):
                output_file, used_char = render_moviepy_video(st.session_state.plan_data)
                char_label = os.path.basename(used_char) if used_char else "기본"
                st.success(f"🎉 '{char_label}' 캐릭터가 부드럽게 합성된 60초(01:00) 완성본 동영상이 제작되었습니다!")
                
                v_col1, v_col2 = st.columns([1, 2])
                with v_col1:
                    st.video(output_file)
                with v_col2:
                    st.markdown("### 📥 영상 다운로드 및 배포")
                    with open(output_file, "rb") as vf:
                        st.download_button(
                            label="⬇️ 완성본 MP4 비디오 다운로드 (60초)",
                            data=vf,
                            file_name="nie_shortform_60s_seamless.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

with tab2:
    st.header("🌐 국립생태원 연구성과 웹페이지 원클릭 팝업 연계")
    st.markdown("""
    국립생태원 홈페이지 [연구성과 게시판](https://www.nie.re.kr/nie/pgm/achieve/thesisList.do?menuNo=200063)에 
    게시글을 등록할 때, 사용자가 **[🎬 숏폼 영상 보기]** 버튼을 누르면 별도 페이지 이동 없이 **모바일 숏폼 팝업 플레이어**가 바로 실행되는 반응형 코드입니다.
    """)
    t_val = st.session_state.plan_data.get("title", "")
    title_text = t_val if t_val else "국립생태원 연구성과"
    s_val = st.session_state.plan_data.get("research_summary", "")
    summary_text = s_val if s_val else "연구 요약 내용"

    html_template = """
<div class="nie-shortform-container" style="margin: 20px 0; font-family: sans-serif;">
    <div style="background: linear-gradient(135deg, #134e2e 0%, #1e3a29 100%); padding: 18px 24px; border-radius: 12px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div>
            <span style="background: #a3e635; color: #0f381e; font-weight: bold; font-size: 12px; padding: 4px 10px; border-radius: 20px;">60초 숏폼</span>
            <h4 style="color: #ffffff; margin: 8px 0 4px 0; font-size: 16px;">__TITLE__</h4>
            <p style="color: #cbd5e1; margin: 0; font-size: 13px;">__SUMMARY__</p>
        </div>
        <button onclick="document.getElementById('nieVideoModal').style.display='flex'" style="background: #a3e635; color: #0f381e; border: none; padding: 12px 22px; border-radius: 30px; font-weight: bold; font-size: 14px; cursor: pointer; transition: 0.2s; white-space: nowrap; margin-left: 15px;">
            ▶ 숏폼 영상 보기
        </button>
    </div>
    <div id="nieVideoModal" style="display: none; position: fixed; z-index: 99999; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.75); justify-content: center; align-items: center;">
        <div style="position: relative; width: 380px; max-width: 90%; background: #000; border-radius: 20px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
            <button onclick="document.getElementById('nieVideoModal').style.display='none'; document.getElementById('nieShortVideo').pause();" style="position: absolute; right: 15px; top: 15px; z-index: 10; background: rgba(0,0,0,0.6); color: #fff; border: none; font-size: 20px; width: 36px; height: 36px; border-radius: 50%; cursor: pointer;">✕</button>
            <video id="nieShortVideo" controls playsinline autoplay style="width: 100%; aspect-ratio: 9/16; display: block;">
                <source src="nie_shortform_60s_seamless.mp4" type="video/mp4">
            </video>
        </div>
    </div>
</div>
"""
    auto_modal_html = html_template.replace("__TITLE__", title_text).replace("__SUMMARY__", summary_text).strip()

    st.subheader("📋 게시판 등록용 소스코드 (HTML 모드 복사)")
    st.code(auto_modal_html, language="html")

    st.markdown("---")
    st.subheader("📱 홈페이지 삽입 시 작동 미리보기")
    st.components.v1.html(auto_modal_html, height=220)

with tab3:
    st.header("✨ 차세대 생성형 AI 비디오 확장 프롬프트 (Google Vids / Runway / Sora)")
    st.markdown("""
    자체 렌더링된 숏폼 영상 외에, **Google Vids, Runway Gen-3, Luma Dream Machine, Kling** 등 최신 AI 비디오 생성 도구에 바로 활용할 수 있는 **방송 다큐멘터리급 시네마틱 프롬프트**입니다.
    """)

    scenes_data = st.session_state.plan_data.get("scenes", [])
    
    st.subheader("🎬 씬별 시네마틱 프롬프트 (원클릭 복사)")
    p_cols = st.columns(5)
    
    all_prompts_text = []
    
    for idx, col in enumerate(p_cols):
        sc = scenes_data[idx] if idx < len(scenes_data) else {}
        cap = sc.get("caption", f"Scene {idx+1}")
        v_prompt = sc.get("visual_prompt_en", "")
        
        if not v_prompt:
            v_prompt = f"Cinematic 8K documentary footage of {scene_headers[idx]}, highly detailed wildlife and ecological scenery, slow camera movement, volumetric natural lighting, photorealistic National Geographic quality, 9:16 vertical format."
        
        all_prompts_text.append(f"[{scene_headers[idx]} - {cap}]\n{v_prompt}")
        
        with col:
            # 1~4단계 탭과 완전히 동일하게 'Scene X (Hook/Problem/Research/Finding/Impact)' 형태로 표기
            st.markdown(f"**{scene_headers[idx]}**")
            st.caption(f"핵심 키워드: {cap}")
            st.code(v_prompt, language="text")

    st.markdown("---")
    st.subheader("📋 5개 씬 전체 프롬프트 일괄 복사")
    st.code("\n\n".join(all_prompts_text), language="text")